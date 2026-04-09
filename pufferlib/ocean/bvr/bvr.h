// Originally made by Sam Turner and Finlay Sanders, 2025.
// Included in pufferlib under the original project's MIT license.
// https://github.com/tensaur/drone

#pragma once

#include <math.h>
#include <limits.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "planelib.h"
#include "missilelib.h"

#define OBS_DIM 18

typedef struct Client Client;
typedef struct Log Log;
typedef struct BvrEnv BvrEnv;
typedef struct BvrEnv Bvr;

struct Log {
  float episode_return;
  float episode_length;
  float perf;
  float oob;
  float timeout;
  float n;
};

struct BvrEnv {
  float *observations;
  float *actions;
  float *rewards;
  unsigned char *terminals;
  Log log;
  int tick;
  int num_agents;
  int framestack;
  float *obs_history;
  Plane *agents;
  Missile missile;
  bool missile_active;
  float missile_spawn_delay_sec;
  int next_missile_spawn_tick;
  Client *client;
};

void init(BvrEnv *env) {
  env->agents = (Plane*) calloc(env->num_agents, sizeof(Plane));
  env->log = (Log){0};
  env->tick = 0;
  env->framestack = env->framestack > 0 ? env->framestack : 1;
  env->obs_history = (float*) calloc(
      env->num_agents * env->framestack * OBS_DIM, sizeof(float));
  env->missile_spawn_delay_sec = 1.0f;
}

void add_log(BvrEnv *env, int idx,
             bool timeout, bool oob) {
  Plane *agent = &env->agents[idx];
  env->log.episode_return += agent->episode_return;
  env->log.episode_length += agent->episode_length;
  env->log.perf += env->rewards[idx];
  if (oob) { env->log.oob += 1.0f; }
  if (timeout) {
    env->log.timeout += 1.0f;
  }
  env->log.n += 1.0f;
  agent->episode_length = 0;
  agent->episode_return = 0.0f;
}

void compute_observations(BvrEnv *env) {
  for (int i = 0; i < env->num_agents; i++) {
    Plane *agent = &env->agents[i];
    float current_obs[OBS_DIM];
    float roll, pitch, yaw;
    quat_to_euler(agent->state.quat, &roll, &pitch, &yaw);

    float speed = norm3(agent->state.vel);
    float mach = clampf(2.0f * (speed / agent->params.max_vel) - 1.0f, -1.0f, 1.0f);
    float grid_x = clampf(agent->state.pos.x / GRID_X, -1.0f, 1.0f);
    float grid_y = clampf(agent->state.pos.y / GRID_Y, -1.0f, 1.0f);
    float grid_z = clampf(agent->state.pos.z / GRID_Z, -1.0f, 1.0f);
    float altitude = clampf(2.0f * (agent->state.pos.z / GRID_Z) - 1.0f, -1.0f, 1.0f);
    float missile_bearing_sin = 0.0f;
    float missile_bearing_cos = 0.0f;
    float missile_heading_sin = 0.0f;
    float missile_heading_cos = 0.0f;
    float missile_mach = 0.0f;
    float missile_altitude = 0.0f;
    float missile_distance = 0.0f;

    if (env->missile_active) {
      float missile_roll, missile_pitch, missile_yaw;
      Vec3 missile_delta = sub3(env->missile.state.pos, agent->state.pos);
      float missile_bearing = atan2f(missile_delta.y, missile_delta.x);
      float missile_speed = norm3(env->missile.state.vel);
      float max_missile_distance =
          sqrtf(4.0f * GRID_X * GRID_X +
                4.0f * GRID_Y * GRID_Y +
                GRID_Z * GRID_Z);

      quat_to_euler(
          env->missile.state.quat, &missile_roll, &missile_pitch, &missile_yaw);

      missile_bearing_sin = sinf(missile_bearing);
      missile_bearing_cos = cosf(missile_bearing);
      missile_heading_sin = sinf(missile_yaw);
      missile_heading_cos = cosf(missile_yaw);
      missile_mach = clampf(
          2.0f * (missile_speed / env->missile.params.max_vel) - 1.0f,
          -1.0f, 1.0f);
      missile_altitude = clampf(
          2.0f * (env->missile.state.pos.z / GRID_Z) - 1.0f,
          -1.0f, 1.0f);
      missile_distance = clampf(
          2.0f * (norm3(missile_delta) / max_missile_distance) - 1.0f,
          -1.0f, 1.0f);
    }

    // BVR observation layout:
    // [0] sin(bearing to target / enemy in world frame)
    // [1] cos(bearing to target / enemy in world frame)
    // [2] sin(own heading / yaw in world frame)
    // [3] cos(own heading / yaw in world frame)
    // [4] own speed proxy ("mach"), derived from world-frame velocity magnitude
    // [5] own altitude from world-frame z position
    // [6] own grid x position in world frame, normalized by GRID_X
    // [7] own grid y position in world frame, normalized by GRID_Y
    // [8] own grid z position in world frame, normalized by GRID_Z
    // [9] sin(enemy bearing to ownship in world frame)
    // [10] cos(enemy bearing to ownship in world frame)
    // [11] sin(enemy heading / yaw in world frame)
    // [12] cos(enemy heading / yaw in world frame)
    // [13] enemy speed proxy ("mach"), derived from world-frame velocity magnitude
    // [14] enemy altitude from world-frame z position
    // [15] own_missile_active
    // [16] enemy_missile_active
    // [17] enemy distance from ownship
    current_obs[0] = 0.0f;
    current_obs[1] = 0.0f;
    current_obs[2] = sinf(yaw);
    current_obs[3] = cosf(yaw);
    current_obs[4] = mach;
    current_obs[5] = altitude;
    current_obs[6] = grid_x;
    current_obs[7] = grid_y;
    current_obs[8] = grid_z;
    current_obs[9] = missile_bearing_sin;
    current_obs[10] = missile_bearing_cos;
    current_obs[11] = missile_heading_sin;
    current_obs[12] = missile_heading_cos;
    current_obs[13] = missile_mach;
    current_obs[14] = missile_altitude;
    current_obs[15] = 0.0f;
    current_obs[16] = env->missile_active ? 1.0f : 0.0f;
    current_obs[17] = missile_distance;

    int history_offset = i * env->framestack * OBS_DIM;
    int stacked_obs_size = env->framestack * OBS_DIM;
    float *history = &env->obs_history[history_offset];
    float *stacked_obs = &env->observations[history_offset];

    if (env->framestack > 1) {
      memmove(
          history,
          history + OBS_DIM,
          sizeof(float) * OBS_DIM * (env->framestack - 1));
    }
    memcpy(
        history + OBS_DIM * (env->framestack - 1),
        current_obs,
        sizeof(float) * OBS_DIM);
    memcpy(stacked_obs, history, sizeof(float) * stacked_obs_size);
  }
}

void clear_observation_history(BvrEnv *env, int idx) {
  memset(
      &env->obs_history[idx * env->framestack * OBS_DIM],
      0,
      sizeof(float) * env->framestack * OBS_DIM);
}

void reset_agent(BvrEnv *env, Plane *agent, int idx) {
  agent->episode_return = 0.0f;
  agent->episode_length = 0;
  float dr = rndf(0.1f, 0.4f);
  init_plane(agent, dr);

  agent->state.pos =
      (Vec3){rndf(-30.0f, 30.0f), rndf(-30.0f, 30.0f),
             rndf(20.0f, 80.0f)};

  agent->prev_pos = agent->state.pos;
  clear_observation_history(env, idx);
}

void c_reset(BvrEnv *env) {
  env->tick = 0;
  init_missile(&env->missile);
  env->missile_active = false;
  env->next_missile_spawn_tick = (int)(env->missile_spawn_delay_sec / DT);

  for (int i = 0; i < env->num_agents; i++) {
    Plane *agent = &env->agents[i];
    reset_agent(env, agent, i);
  }

  compute_observations(env);
}

void c_step(BvrEnv *env) {
  env->tick += 1;

  if (!env->missile_active && env->tick >= env->next_missile_spawn_tick) {
    init_missile(&env->missile);
    env->missile.state.pos = (Vec3){0.0f, 0.0f, 1.0f};
    if (env->num_agents > 0) {
      aim_missile_at_point(&env->missile, env->agents[0].state.pos, 50.0f);
    }
    env->missile_active = true;
  }

  for (int i = 0; i < env->num_agents; i++) {
    Plane *agent = &env->agents[i];
    env->rewards[i] = 0;
    env->terminals[i] = 0;

    float *atn = &env->actions[3 * i];
    move_aircraft(agent, atn);

    bool out_of_bounds =
        agent->state.pos.x < -GRID_X || agent->state.pos.x > GRID_X ||
        agent->state.pos.y < -GRID_Y || agent->state.pos.y > GRID_Y ||
        agent->state.pos.z < 0.0f || agent->state.pos.z > GRID_Z;

    float reward = 0.0f;

    // Update agent state
    agent->episode_length++;

    env->rewards[i] = reward;
    agent->episode_return += reward;

    // Check termination conditions
    if (out_of_bounds) {
      env->rewards[i] -= 1.0f;
      agent->episode_return -= 1.0f;
      env->terminals[i] = 1;
      add_log(env, i,false, true);
      c_reset(env);
      return;
    } else if (env->tick >= HORIZON - 1) {
      env->rewards[i] += 1.0f;
      agent->episode_return += 1.0f;
      env->terminals[i] = 1;
      add_log(env, i, true,false);
      c_reset(env);
      return;
    }
  }

  if (env->missile_active && env->num_agents > 0) {
    move_missile(&env->missile, env->agents[0].state.pos);

    Vec3 missile_to_agent = sub3(env->agents[0].state.pos, env->missile.state.pos);
    float missile_distance = norm3(missile_to_agent);
    bool missile_hit_agent = missile_distance < 3.0f;

    bool missile_out_of_bounds =
        env->missile.state.pos.x < -GRID_X || env->missile.state.pos.x > GRID_X ||
        env->missile.state.pos.y < -GRID_Y || env->missile.state.pos.y > GRID_Y ||
        env->missile.state.pos.z < 0.0f || env->missile.state.pos.z > GRID_Z;

    if (missile_hit_agent) {
      env->missile_active = false;
      env->next_missile_spawn_tick = env->tick + (int)(env->missile_spawn_delay_sec / DT);
      env->rewards[0] -= 1.0f;
      env->agents[0].episode_return -= 1.0f;
      env->terminals[0] = 1;
      add_log(env, 0, false, false);
      c_reset(env);
      return;
    } else if (missile_out_of_bounds) {
      env->missile_active = false;
      env->next_missile_spawn_tick = env->tick + (int)(env->missile_spawn_delay_sec / DT);
    }
  }

  compute_observations(env);
}

void c_close_client(Client* client);

void c_close(BvrEnv *env) {
  free(env->agents);
  free(env->obs_history);

  if (env->client != NULL) {
    c_close_client(env->client);
  }
}
