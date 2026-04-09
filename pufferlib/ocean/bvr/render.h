// Originally made by Sam Turner and Finlay Sanders, 2025.
// Included in pufferlib under the original project's MIT license.
// https://github.com/tensaur/drone

#pragma once

#include <math.h>

#include "bvr.h"
#include "raylib.h"

#define R (Color){255, 0, 0, 255}
#define W (Color){255, 255, 255, 255}
#define B (Color){0, 0, 255, 255}
Color COLORS[64] = {
    W, B, B, R, R, B, B, W,
    B, W, B, R, R, B, W, B,
    B, B, W, R, R, W, B, B,
    R, R, R, R, R, R, R, R,
    R, R, R, R, R, R, R, R,
    B, B, W, R, R, W, B, B,
    B, W, B, R, R, B, W, B,
    W, B, B, R, R, B, B, W
};
#undef R
#undef W
#undef B

typedef struct Client Client;

struct Client {
  Camera3D camera;
  float width;
  float height;

  float camera_distance;
  float camera_azimuth;
  float camera_elevation;
  bool is_dragging;
  Vector2 last_mouse_pos;

  // Trailing path buffer (for rendering only)
  Trail *trails;
  Trail missile_trail;
};

void c_close_client(Client *client) {
  CloseWindow();
  free(client->trails);
  free(client);
}

static void update_camera_position(Client *c) {
  float r = c->camera_distance;
  float az = c->camera_azimuth;
  float el = c->camera_elevation;

  float x = r * cosf(el) * cosf(az);
  float y = r * cosf(el) * sinf(az);
  float z = r * sinf(el);

  c->camera.position = (Vector3){x, y, z};
  c->camera.target = (Vector3){0, 0, GRID_Z * 0.5f};
}

void handle_camera_controls(Client *client) {
  Vector2 mouse_pos = GetMousePosition();

  if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
    client->is_dragging = true;
    client->last_mouse_pos = mouse_pos;
  }

  if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
    client->is_dragging = false;
  }

  if (client->is_dragging && IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
    Vector2 mouse_delta = {mouse_pos.x - client->last_mouse_pos.x,
                           mouse_pos.y - client->last_mouse_pos.y};

    float sensitivity = 0.005f;

    client->camera_azimuth -= mouse_delta.x * sensitivity;

    client->camera_elevation += mouse_delta.y * sensitivity;
    client->camera_elevation =
        clampf(client->camera_elevation, -PI / 2.0f + 0.1f, PI / 2.0f - 0.1f);

    client->last_mouse_pos = mouse_pos;

    update_camera_position(client);
  }

  float wheel = GetMouseWheelMove();
  if (wheel != 0) {
    client->camera_distance -= wheel * 2.0f;
    client->camera_distance = clampf(client->camera_distance, 5.0f, 300.0f);
    update_camera_position(client);
  }
}

Client *make_client(BvrEnv *env) {
  Client *client = (Client *)calloc(1, sizeof(Client));

  client->width = WIDTH;
  client->height = HEIGHT;

  SetConfigFlags(FLAG_MSAA_4X_HINT); // antialiasing
  InitWindow(WIDTH, HEIGHT, "PufferLib BVR");

#ifndef __EMSCRIPTEN__
  SetTargetFPS(60);
#endif

  if (!IsWindowReady()) {
    TraceLog(LOG_ERROR, "Window failed to initialize\n");
    free(client);
    return NULL;
  }

  client->camera_distance = 140.0f;
  client->camera_azimuth = 0.0f;
  client->camera_elevation = PI / 7.0f;
  client->is_dragging = false;
  client->last_mouse_pos = (Vector2){0.0f, 0.0f};

  client->camera.up = (Vector3){0.0f, 0.0f, 1.0f};
  client->camera.fovy = 55.0f;
  client->camera.projection = CAMERA_PERSPECTIVE;

  update_camera_position(client);

  // Initialize trail buffer
  client->trails = (Trail *)calloc(env->num_agents, sizeof(Trail));
  for (int i = 0; i < env->num_agents; i++) {
    Trail *trail = &client->trails[i];
    trail->index = 0;
    trail->count = 0;
    for (int j = 0; j < TRAIL_LENGTH; j++) {
      trail->pos[j] = env->agents[i].state.pos;
    }
  }

  client->missile_trail.index = 0;
  client->missile_trail.count = 0;
  for (int j = 0; j < TRAIL_LENGTH; j++) {
    client->missile_trail.pos[j] = (Vec3){0.0f, 0.0f, 0.0f};
  }

  return client;
}

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

void c_render(BvrEnv *env) {
  if (env->client == NULL) {
    env->client = make_client(env);
    if (env->client == NULL) {
      TraceLog(LOG_ERROR, "Failed to initialize client for rendering\n");
      return;
    }
  }

  if (WindowShouldClose()) {
    c_close(env);
    exit(0);
  }

  if (IsKeyDown(KEY_ESCAPE)) {
    c_close(env);
    exit(0);
  }

  handle_camera_controls(env->client);

  Client *client = env->client;

  for (int i = 0; i < env->num_agents; i++) {
    Plane *agent = &env->agents[i];
    Trail *trail = &client->trails[i];
    trail->pos[trail->index] = agent->state.pos;
    trail->index = (trail->index + 1) % TRAIL_LENGTH;
    if (trail->count < TRAIL_LENGTH) {
      trail->count++;
    }
    if (env->terminals[i]) {
      trail->index = 0;
      trail->count = 0;
    }
  }

  if (env->missile_active) {
    Trail *trail = &client->missile_trail;
    trail->pos[trail->index] = env->missile.state.pos;
    trail->index = (trail->index + 1) % TRAIL_LENGTH;
    if (trail->count < TRAIL_LENGTH) {
      trail->count++;
    }
  } else {
    client->missile_trail.index = 0;
    client->missile_trail.count = 0;
  }

  BeginDrawing();
  ClearBackground(PUFF_BACKGROUND);

  BeginMode3D(client->camera);

  // Draw the playable volume with ground at z=0 and ceiling at z=GRID_Z.
  DrawCubeWires((Vector3){0.0f, 0.0f, GRID_Z * 0.5f}, GRID_X * 2.0f, GRID_Y * 2.0f,
                GRID_Z, WHITE);

  // Draw a ground outline on the z=0 plane.
  DrawLine3D((Vector3){-GRID_X, -GRID_Y, 0.0f}, (Vector3){ GRID_X, -GRID_Y, 0.0f}, GRAY);
  DrawLine3D((Vector3){ GRID_X, -GRID_Y, 0.0f}, (Vector3){ GRID_X,  GRID_Y, 0.0f}, GRAY);
  DrawLine3D((Vector3){ GRID_X,  GRID_Y, 0.0f}, (Vector3){-GRID_X,  GRID_Y, 0.0f}, GRAY);
  DrawLine3D((Vector3){-GRID_X,  GRID_Y, 0.0f}, (Vector3){-GRID_X, -GRID_Y, 0.0f}, GRAY);

  for (int i = 0; i < env->num_agents; i++) {
    Plane *agent = &env->agents[i];

    // Draw a simple oriented aircraft marker: nose, wings, and tail.
    Color body_color = COLORS[i];
    Vec3 pos = agent->state.pos;
    Vec3 nose = quat_rotate(agent->state.quat, (Vec3){3.0f, 0.0f, 0.0f});
    Vec3 left_wing = quat_rotate(agent->state.quat, (Vec3){0.0f, 1.5f, 0.0f});
    Vec3 right_wing = quat_rotate(agent->state.quat, (Vec3){0.0f, -1.5f, 0.0f});
    Vec3 tail = quat_rotate(agent->state.quat, (Vec3){-1.5f, 0.0f, 0.0f});
    Vec3 tail_fin = quat_rotate(agent->state.quat, (Vec3){-1.2f, 0.0f, 0.8f});

    Vector3 center = {pos.x, pos.y, pos.z};
    Vector3 nose_tip = {pos.x + nose.x, pos.y + nose.y, pos.z + nose.z};
    Vector3 left_tip = {pos.x + left_wing.x, pos.y + left_wing.y, pos.z + left_wing.z};
    Vector3 right_tip = {pos.x + right_wing.x, pos.y + right_wing.y, pos.z + right_wing.z};
    Vector3 tail_tip = {pos.x + tail.x, pos.y + tail.y, pos.z + tail.z};
    Vector3 fin_tip = {pos.x + tail_fin.x, pos.y + tail_fin.y, pos.z + tail_fin.z};

    DrawSphere(center, 0.35f, body_color);
    DrawLine3D(center, nose_tip, body_color);
    DrawLine3D(left_tip, right_tip, body_color);
    DrawLine3D(center, tail_tip, body_color);
    DrawLine3D(tail_tip, fin_tip, body_color);

    // draws line with direction and magnitude of velocity / 10
    if (norm3(agent->state.vel) > 0.1f) {
      DrawLine3D(
          (Vector3){agent->state.pos.x, agent->state.pos.y, agent->state.pos.z},
          (Vector3){agent->state.pos.x + agent->state.vel.x * 0.1f,
                    agent->state.pos.y + agent->state.vel.y * 0.1f,
                    agent->state.pos.z + agent->state.vel.z * 0.1f},
          MAGENTA);
    }

    // Draw trailing path
    Trail *trail = &client->trails[i];
    if (trail->count <= 2) {
      continue;
    }
    for (int j = 0; j < trail->count - 1; j++) {
      int idx0 = (trail->index - j - 1 + TRAIL_LENGTH) % TRAIL_LENGTH;
      int idx1 = (trail->index - j - 2 + TRAIL_LENGTH) % TRAIL_LENGTH;
      float alpha =
          (float)(TRAIL_LENGTH - j) / (float)trail->count * 0.8f; // fade out
      Color trail_color = ColorAlpha((Color){0, 187, 187, 255}, alpha);
      DrawLine3D(
          (Vector3){trail->pos[idx0].x, trail->pos[idx0].y, trail->pos[idx0].z},
          (Vector3){trail->pos[idx1].x, trail->pos[idx1].y, trail->pos[idx1].z},
          trail_color);
    }
  }

  if (env->missile_active) {
    Trail *trail = &client->missile_trail;
    for (int j = 0; j < trail->count - 1; j++) {
      int idx0 = (trail->index - j - 1 + TRAIL_LENGTH) % TRAIL_LENGTH;
      int idx1 = (trail->index - j - 2 + TRAIL_LENGTH) % TRAIL_LENGTH;
      float alpha =
          (float)(TRAIL_LENGTH - j) / (float)trail->count * 0.9f;
      Color trail_color = ColorAlpha(RED, alpha);
      DrawLine3D(
          (Vector3){trail->pos[idx0].x, trail->pos[idx0].y, trail->pos[idx0].z},
          (Vector3){trail->pos[idx1].x, trail->pos[idx1].y, trail->pos[idx1].z},
          trail_color);
    }

    DrawSphere(
        (Vector3){env->missile.state.pos.x, env->missile.state.pos.y, env->missile.state.pos.z},
        0.5f, RED);
  }

  EndMode3D();

  DrawText("Left click + drag: Rotate camera", 10, 10, 16, PUFF_WHITE);
  DrawText("Mouse wheel: Zoom in/out", 10, 30, 16, PUFF_WHITE);

  EndDrawing();
}
