// Standalone C demo for bvr environment
// Compile using: ./scripts/build_ocean.sh bvr [local|fast]
// Run with: ./bvr

#include "bvr.h"
#include "render.h"
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <errno.h>

typedef struct {
    bool use_preset;
    int preset;
    bool use_turn_sequence;
    int sequence_period;
    float heading_action;
    float altitude_action;
    float throttle_action;
    float pos_x;
    float pos_y;
    float pos_z;
    float pitch;
    float speed;
    float throttle_state;
    float k_heading;
    float max_bank_cmd;
    float k_bank;
    float k_roll_rate;
    float k_altitude;
    float max_pitch_cmd;
    float k_pitch;
    float k_pitch_rate;
    float k_beta;
    float k_yaw_rate;
} TrimConfig;

typedef struct {
    int terminated;
    int step;
    float mean_abs_roll;
    float mean_abs_pitch;
    float mean_abs_omega;
    float max_abs_roll;
    float max_abs_pitch;
    float max_abs_omega;
    float altitude_drift;
    float speed_drift;
    float score;
} TrimSummary;

typedef struct {
    double sum_abs_roll;
    double sum_abs_pitch;
    double sum_abs_omega;
    float max_abs_roll;
    float max_abs_pitch;
    float max_abs_omega;
    float initial_z;
    float initial_speed;
    int samples;
} TrimStats;

static float parse_float_arg(const char *label, const char *value) {
    char *end = NULL;
    errno = 0;
    float parsed = strtof(value, &end);
    if (errno != 0 || end == value || *end != '\0') {
        fprintf(stderr, "ERROR: Invalid %s value: %s\n", label, value);
        exit(1);
    }
    return parsed;
}

static void init_trim_config(TrimConfig *cfg) {
    *cfg = (TrimConfig){
        .use_preset = true,
        .preset = 0,
        .use_turn_sequence = false,
        .sequence_period = 120,
        .heading_action = 0.0f,
        .altitude_action = 0.0f,
        .throttle_action = 0.0f,
        .pos_x = 0.0f,
        .pos_y = 0.0f,
        .pos_z = 75.0f,
        .pitch = -0.08f,
        .speed = BASE_INIT_U,
        .throttle_state = BASE_INIT_THROTTLE,
        .k_heading = BASE_K_HEADING,
        .max_bank_cmd = BASE_MAX_BANK_CMD,
        .k_bank = BASE_K_BANK,
        .k_roll_rate = BASE_K_ROLL_RATE,
        .k_altitude = BASE_K_ALTITUDE,
        .max_pitch_cmd = BASE_MAX_PITCH_CMD,
        .k_pitch = BASE_K_PITCH,
        .k_pitch_rate = BASE_K_PITCH_RATE,
        .k_beta = BASE_K_BETA,
        .k_yaw_rate = BASE_K_YAW_RATE,
    };
}

void generate_dummy_actions(BvrEnv *env) {
    // Generate random floats in [-1, 1] range
    env->actions[0] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    env->actions[1] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    env->actions[2] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

void set_fixed_actions(BvrEnv *env, float heading, float altitude, float throttle) {
    env->actions[0] = heading;
    env->actions[1] = altitude;
    env->actions[2] = throttle;
}

void set_preset_actions(BvrEnv *env, int preset) {
    switch (preset) {
        case 0:
            // Neutral hold
            set_fixed_actions(env, 0.0f, 0.0f, 0.0f);
            break;
        case 1:
            // Higher throttle, nominal altitude hold
            set_fixed_actions(env, 0.0f, 0.0f, 1.0f);
            break;
        case 2:
            // Higher altitude target and higher throttle
            set_fixed_actions(env, 0.0f, 1.0f, 1.0f);
            break;
        case 3:
            // Higher altitude target, nominal throttle
            set_fixed_actions(env, 0.0f, 1.0f, 0.0f);
            break;
        case 4:
            // Lower altitude target, higher throttle
            set_fixed_actions(env, 0.0f, -1.0f, 1.0f);
            break;
        default:
            set_fixed_actions(env, 0.0f, 0.0f, 0.0f);
            break;
    }
}

static void set_config_actions(BvrEnv *env, const TrimConfig *cfg) {
    if (cfg->use_turn_sequence) {
        float roll, pitch, yaw;
        quat_to_euler(env->agents[0].state.quat, &roll, &pitch, &yaw);
        (void)roll;
        (void)pitch;
        float target_yaw = yaw + 2.0f * PI / 3.0f;
        while (target_yaw > PI) {
            target_yaw -= 2.0f * PI;
        }
        while (target_yaw < -PI) {
            target_yaw += 2.0f * PI;
        }
        float heading_action = clampf(target_yaw / PI, -1.0f, 1.0f);

        set_fixed_actions(env, heading_action, cfg->altitude_action, cfg->throttle_action);
        return;
    }

    if (cfg->use_preset) {
        set_preset_actions(env, cfg->preset);
        return;
    }

    set_fixed_actions(
        env,
        cfg->heading_action,
        cfg->altitude_action,
        cfg->throttle_action
    );
}

static void apply_trim_config(BvrEnv *env, const TrimConfig *cfg) {
    Plane *agent = &env->agents[0];
    agent->state.pos = (Vec3){cfg->pos_x, cfg->pos_y, cfg->pos_z};
    agent->state.quat = quat_from_euler(0.0f, cfg->pitch, 0.0f);
    agent->state.vel = quat_rotate(
        agent->state.quat, (Vec3){cfg->speed, 0.0f, 0.0f}
    );
    agent->state.omega = (Vec3){0.0f, 0.0f, 0.0f};
    agent->state.throttle_state = clampf(cfg->throttle_state, 0.0f, 1.0f);
    agent->prev_pos = agent->state.pos;

    agent->control_params.k_heading = cfg->k_heading;
    agent->control_params.max_bank_cmd = cfg->max_bank_cmd;
    agent->control_params.k_bank = cfg->k_bank;
    agent->control_params.k_roll_rate = cfg->k_roll_rate;
    agent->control_params.k_altitude = cfg->k_altitude;
    agent->control_params.max_pitch_cmd = cfg->max_pitch_cmd;
    agent->control_params.k_pitch = cfg->k_pitch;
    agent->control_params.k_pitch_rate = cfg->k_pitch_rate;
    agent->control_params.k_beta = cfg->k_beta;
    agent->control_params.k_yaw_rate = cfg->k_yaw_rate;
}

static void print_trim_config(const TrimConfig *cfg, const char *csv_path, int steps) {
    printf(
        "trim_config steps=%d csv=%s action=(%.3f %.3f %.3f) "
        "pos=(%.3f %.3f %.3f) pitch=%.4f speed=%.3f throttle_state=%.3f "
        "gains=(k_heading=%.3f max_bank=%.3f k_bank=%.3f k_roll_rate=%.3f "
        "k_altitude=%.3f max_pitch=%.3f k_pitch=%.3f k_pitch_rate=%.3f "
        "k_beta=%.3f k_yaw_rate=%.3f)\n",
        steps,
        csv_path,
        cfg->heading_action, cfg->altitude_action, cfg->throttle_action,
        cfg->pos_x, cfg->pos_y, cfg->pos_z,
        cfg->pitch, cfg->speed, cfg->throttle_state,
        cfg->k_heading, cfg->max_bank_cmd, cfg->k_bank, cfg->k_roll_rate,
        cfg->k_altitude, cfg->max_pitch_cmd, cfg->k_pitch, cfg->k_pitch_rate,
        cfg->k_beta, cfg->k_yaw_rate
    );
}

void test_performance(int timeout) {
    srand(time(NULL)); // Seed random number generator

    BvrEnv *env = calloc(1, sizeof(BvrEnv));
    env->num_agents = 1;
    env->framestack = 1;
    init(env);

    size_t obs_size = OBS_DIM * env->framestack;
    size_t act_size = 3;
    env->observations = (float *)calloc(env->num_agents * obs_size, sizeof(float));
    env->actions = (float *)calloc(env->num_agents * act_size, sizeof(float));
    env->rewards = (float *)calloc(env->num_agents, sizeof(float));
    env->terminals = (unsigned char *)calloc(env->num_agents, sizeof(unsigned char));

    if (!env->observations || !env->actions || !env->rewards) {
        fprintf(stderr, "ERROR: Failed to allocate memory for demo buffers.\n");
        free(env->observations);
        free(env->actions);
        free(env->rewards);
        free(env->terminals);
        free(env);
        return;
    }

    c_reset(env);

    int start = time(NULL);
    int num_steps = 0;
    while (time(NULL) - start < timeout) {
        generate_dummy_actions(env);
        c_step(env);
        num_steps++;
    }

    int end = time(NULL);
    float sps = (env->num_agents * num_steps) / (float)(end - start);
    printf("Test Environment SPS: %f\n", sps);

    c_close(env);
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free(env);
}

void print_state_summary(BvrEnv *env, int step) {
    Plane *agent = &env->agents[0];
    float roll, pitch, yaw;
    quat_to_euler(agent->state.quat, &roll, &pitch, &yaw);
    float speed = norm3(agent->state.vel);
    float missile_roll = 0.0f, missile_pitch = 0.0f, missile_yaw = 0.0f;
    float missile_speed = 0.0f;
    float missile_distance = -1.0f;

    if (env->missile_active) {
        quat_to_euler(env->missile.state.quat, &missile_roll, &missile_pitch, &missile_yaw);
        missile_speed = norm3(env->missile.state.vel);
        missile_distance = norm3(sub3(agent->state.pos, env->missile.state.pos));
    }

    printf(
        "step=%d pos=(%.2f %.2f %.2f) vel=(%.2f %.2f %.2f) speed=%.2f "
        "rpy=(%.3f %.3f %.3f) omega=(%.3f %.3f %.3f) "
        "missile_active=%d missile_pos=(%.2f %.2f %.2f) missile_vel=(%.2f %.2f %.2f) "
        "missile_speed=%.2f missile_rpy=(%.3f %.3f %.3f) missile_dist=%.2f "
        "reward=%.3f done=%d\n",
        step,
        agent->state.pos.x, agent->state.pos.y, agent->state.pos.z,
        agent->state.vel.x, agent->state.vel.y, agent->state.vel.z,
        speed,
        roll, pitch, yaw,
        agent->state.omega.x, agent->state.omega.y, agent->state.omega.z,
        env->missile_active ? 1 : 0,
        env->missile.state.pos.x, env->missile.state.pos.y, env->missile.state.pos.z,
        env->missile.state.vel.x, env->missile.state.vel.y, env->missile.state.vel.z,
        missile_speed,
        missile_roll, missile_pitch, missile_yaw,
        missile_distance,
        env->rewards[0], env->terminals[0]
    );
}

void print_control_summary(BvrEnv *env, int step) {
    Plane *agent = &env->agents[0];
    Controls cmd;
    action_to_controls(
        &agent->state,
        &agent->control_params,
        env->actions,
        &cmd
    );

    printf(
        "controls step=%d throttle=%.3f aileron=%.3f elevator=%.3f rudder=%.3f action=(%.3f %.3f %.3f)\n",
        step,
        cmd.throttle, cmd.aileron, cmd.elevator, cmd.rudder,
        env->actions[0], env->actions[1], env->actions[2]
    );
}

void write_state_csv(FILE *fp, BvrEnv *env, int step) {
    Plane *agent = &env->agents[0];
    float roll, pitch, yaw;
    quat_to_euler(agent->state.quat, &roll, &pitch, &yaw);
    float speed = norm3(agent->state.vel);
    float missile_roll = 0.0f, missile_pitch = 0.0f, missile_yaw = 0.0f;
    float missile_speed = 0.0f;
    float missile_distance = -1.0f;

    if (env->missile_active) {
        quat_to_euler(env->missile.state.quat, &missile_roll, &missile_pitch, &missile_yaw);
        missile_speed = norm3(env->missile.state.vel);
        missile_distance = norm3(sub3(agent->state.pos, env->missile.state.pos));
    }

    fprintf(
        fp,
        "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,"
        "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
        step,
        agent->state.pos.x, agent->state.pos.y, agent->state.pos.z,
        agent->state.vel.x, agent->state.vel.y, agent->state.vel.z,
        speed,
        roll, pitch, yaw,
        agent->state.omega.x, agent->state.omega.y, agent->state.omega.z,
        env->terminals[0],
        env->missile_active ? 1 : 0,
        env->missile.state.pos.x, env->missile.state.pos.y, env->missile.state.pos.z,
        env->missile.state.vel.x, env->missile.state.vel.y, env->missile.state.vel.z,
        missile_speed,
        missile_roll, missile_pitch, missile_yaw,
        missile_distance
    );
}

static void update_trim_stats(TrimStats *stats, BvrEnv *env) {
    Plane *agent = &env->agents[0];
    float roll, pitch, yaw;
    quat_to_euler(agent->state.quat, &roll, &pitch, &yaw);
    (void)yaw;
    float abs_roll = fabsf(roll);
    float abs_pitch = fabsf(pitch);
    float max_abs_omega = fmaxf(
        fabsf(agent->state.omega.x),
        fmaxf(fabsf(agent->state.omega.y), fabsf(agent->state.omega.z))
    );

    stats->sum_abs_roll += abs_roll;
    stats->sum_abs_pitch += abs_pitch;
    stats->sum_abs_omega += max_abs_omega;
    stats->max_abs_roll = fmaxf(stats->max_abs_roll, abs_roll);
    stats->max_abs_pitch = fmaxf(stats->max_abs_pitch, abs_pitch);
    stats->max_abs_omega = fmaxf(stats->max_abs_omega, max_abs_omega);
    stats->samples += 1;
}

static TrimSummary finalize_trim_summary(
    TrimStats *stats, BvrEnv *env, int final_step, bool terminated
) {
    Plane *agent = &env->agents[0];
    float final_speed = norm3(agent->state.vel);
    float samples = stats->samples > 0 ? (float)stats->samples : 1.0f;
    TrimSummary summary = {
        .terminated = terminated ? 1 : 0,
        .step = final_step,
        .mean_abs_roll = (float)(stats->sum_abs_roll / samples),
        .mean_abs_pitch = (float)(stats->sum_abs_pitch / samples),
        .mean_abs_omega = (float)(stats->sum_abs_omega / samples),
        .max_abs_roll = stats->max_abs_roll,
        .max_abs_pitch = stats->max_abs_pitch,
        .max_abs_omega = stats->max_abs_omega,
        .altitude_drift = fabsf(agent->state.pos.z - stats->initial_z),
        .speed_drift = fabsf(final_speed - stats->initial_speed),
        .score = 0.0f,
    };

    summary.score =
        5.0f * summary.terminated
        + summary.mean_abs_roll
        + summary.mean_abs_pitch
        + 0.25f * summary.mean_abs_omega
        + 0.5f * summary.max_abs_roll
        + 0.5f * summary.max_abs_pitch
        + 0.05f * summary.max_abs_omega
        + 0.02f * summary.altitude_drift
        + 0.02f * summary.speed_drift;

    return summary;
}

static void print_trim_summary(const TrimSummary *summary) {
    printf(
        "trim_summary terminated=%d step=%d score=%.6f "
        "mean_abs_roll=%.6f mean_abs_pitch=%.6f mean_abs_omega=%.6f "
        "max_abs_roll=%.6f max_abs_pitch=%.6f max_abs_omega=%.6f "
        "altitude_drift=%.6f speed_drift=%.6f\n",
        summary->terminated,
        summary->step,
        summary->score,
        summary->mean_abs_roll,
        summary->mean_abs_pitch,
        summary->mean_abs_omega,
        summary->max_abs_roll,
        summary->max_abs_pitch,
        summary->max_abs_omega,
        summary->altitude_drift,
        summary->speed_drift
    );
}

void test_fixed_stability(int steps, const TrimConfig *cfg, const char *csv_path) {
    srand(0);

    BvrEnv *env = calloc(1, sizeof(BvrEnv));
    env->num_agents = 1;
    env->framestack = 1;

    size_t obs_size = OBS_DIM * env->framestack;
    size_t act_size = 3;
    env->observations = (float *)calloc(env->num_agents * obs_size, sizeof(float));
    env->actions = (float *)calloc(env->num_agents * act_size, sizeof(float));
    env->rewards = (float *)calloc(env->num_agents, sizeof(float));
    env->terminals = (unsigned char *)calloc(env->num_agents, sizeof(unsigned char));

    if (!env->observations || !env->actions || !env->rewards || !env->terminals) {
        fprintf(stderr, "ERROR: Failed to allocate memory for stability buffers.\n");
        free(env->observations);
        free(env->actions);
        free(env->rewards);
        free(env->terminals);
        free(env);
        return;
    }

    init(env);
    c_reset(env);
    apply_trim_config(env, cfg);
    set_config_actions(env, cfg);

    FILE *fp = fopen(csv_path, "w");
    if (!fp) {
        fprintf(stderr, "ERROR: Failed to open csv path: %s\n", csv_path);
        c_close(env);
        free(env->observations);
        free(env->actions);
        free(env->rewards);
        free(env->terminals);
        free(env);
        return;
    }

    fprintf(
        fp,
        "step,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,speed,roll,pitch,yaw,omega_x,omega_y,omega_z,done,"
        "missile_active,missile_pos_x,missile_pos_y,missile_pos_z,missile_vel_x,missile_vel_y,missile_vel_z,"
        "missile_speed,missile_roll,missile_pitch,missile_yaw,missile_distance\n"
    );

    TrimStats stats = {
        .initial_z = env->agents[0].state.pos.z,
        .initial_speed = norm3(env->agents[0].state.vel),
    };

    print_trim_config(cfg, csv_path, steps);
    print_state_summary(env, 0);
    print_control_summary(env, 0);
    write_state_csv(fp, env, 0);
    update_trim_stats(&stats, env);
    int final_step = 0;
    bool terminated = false;
    for (int i = 1; i <= steps; i++) {
        c_step(env);
        set_config_actions(env, cfg);
        write_state_csv(fp, env, i);
        update_trim_stats(&stats, env);
        final_step = i;
        if (i <= 20 || i % 20 == 0 || env->terminals[0]) {
            print_state_summary(env, i);
            if (i <= 10 || env->terminals[0]) {
                print_control_summary(env, i);
            }
        }
        if (env->terminals[0]) {
            terminated = true;
            break;
        }
    }
    fclose(fp);
    TrimSummary summary = finalize_trim_summary(&stats, env, final_step, terminated);
    print_trim_summary(&summary);

    c_close(env);
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free(env);
}

int main(int argc, char **argv) {
    srand(time(NULL)); // Seed random number generator

    if (argc >= 2 && strcmp(argv[1], "headless") == 0) {
        int steps = (argc >= 3) ? atoi(argv[2]) : 400;
        TrimConfig cfg;
        init_trim_config(&cfg);
        const char *csv_path = "/tmp/bvr_headless.csv";

        int argi = 3;
        if (argc >= 4 && argv[3][0] != '-') {
            cfg.preset = atoi(argv[3]);
            argi = 4;
        }
        if (argc >= 5 && argv[4][0] != '-') {
            csv_path = argv[4];
            argi = 5;
        }

        while (argi < argc) {
            const char *arg = argv[argi];
            if (strcmp(arg, "--action") == 0 && argi + 3 < argc) {
                cfg.use_preset = false;
                cfg.heading_action = parse_float_arg("heading_action", argv[argi + 1]);
                cfg.altitude_action = parse_float_arg("altitude_action", argv[argi + 2]);
                cfg.throttle_action = parse_float_arg("throttle_action", argv[argi + 3]);
                argi += 4;
            } else if (strcmp(arg, "--pos") == 0 && argi + 3 < argc) {
                cfg.pos_x = parse_float_arg("pos_x", argv[argi + 1]);
                cfg.pos_y = parse_float_arg("pos_y", argv[argi + 2]);
                cfg.pos_z = parse_float_arg("pos_z", argv[argi + 3]);
                argi += 4;
            } else if (strcmp(arg, "--pitch") == 0 && argi + 1 < argc) {
                cfg.pitch = parse_float_arg("pitch", argv[argi + 1]);
                argi += 2;
            } else if (strcmp(arg, "--speed") == 0 && argi + 1 < argc) {
                cfg.speed = parse_float_arg("speed", argv[argi + 1]);
                argi += 2;
            } else if (strcmp(arg, "--throttle-state") == 0 && argi + 1 < argc) {
                cfg.throttle_state = parse_float_arg("throttle_state", argv[argi + 1]);
                argi += 2;
            } else if (strcmp(arg, "--k-heading") == 0 && argi + 1 < argc) {
                cfg.k_heading = parse_float_arg("k_heading", argv[argi + 1]);
                argi += 2;
            } else if (strcmp(arg, "--max-bank-cmd") == 0 && argi + 1 < argc) {
                cfg.max_bank_cmd = parse_float_arg("max_bank_cmd", argv[argi + 1]);
                argi += 2;
            } else if (strcmp(arg, "--k-bank") == 0 && argi + 1 < argc) {
                cfg.k_bank = parse_float_arg("k_bank", argv[argi + 1]);
                argi += 2;
            } else if (strcmp(arg, "--k-roll-rate") == 0 && argi + 1 < argc) {
                cfg.k_roll_rate = parse_float_arg("k_roll_rate", argv[argi + 1]);
                argi += 2;
            } else if (strcmp(arg, "--k-altitude") == 0 && argi + 1 < argc) {
                cfg.k_altitude = parse_float_arg("k_altitude", argv[argi + 1]);
                argi += 2;
            } else if (strcmp(arg, "--max-pitch-cmd") == 0 && argi + 1 < argc) {
                cfg.max_pitch_cmd = parse_float_arg("max_pitch_cmd", argv[argi + 1]);
                argi += 2;
            } else if (strcmp(arg, "--k-pitch") == 0 && argi + 1 < argc) {
                cfg.k_pitch = parse_float_arg("k_pitch", argv[argi + 1]);
                argi += 2;
            } else if (strcmp(arg, "--k-pitch-rate") == 0 && argi + 1 < argc) {
                cfg.k_pitch_rate = parse_float_arg("k_pitch_rate", argv[argi + 1]);
                argi += 2;
            } else if (strcmp(arg, "--k-beta") == 0 && argi + 1 < argc) {
                cfg.k_beta = parse_float_arg("k_beta", argv[argi + 1]);
                argi += 2;
            } else if (strcmp(arg, "--k-yaw-rate") == 0 && argi + 1 < argc) {
                cfg.k_yaw_rate = parse_float_arg("k_yaw_rate", argv[argi + 1]);
                argi += 2;
            } else if (strcmp(arg, "--turn-seq") == 0) {
                cfg.use_turn_sequence = true;
                cfg.use_preset = false;
                cfg.altitude_action = -0.5f;
                cfg.throttle_action = 0.0f;
                argi += 1;
            } else if (strcmp(arg, "--turn-seq-period") == 0 && argi + 1 < argc) {
                cfg.sequence_period = atoi(argv[argi + 1]);
                argi += 2;
            } else {
                fprintf(stderr, "ERROR: Unknown or incomplete argument: %s\n", arg);
                return 1;
            }
        }

        test_fixed_stability(steps, &cfg, csv_path);
        return 0;
    }

    TrimConfig render_cfg;
    init_trim_config(&render_cfg);
    int argi = 1;
    if (argc >= 2 && strcmp(argv[1], "render") == 0) {
        argi = 2;
    }
    while (argi < argc) {
        const char *arg = argv[argi];
        if (strcmp(arg, "--preset") == 0 && argi + 1 < argc) {
            render_cfg.use_preset = true;
            render_cfg.preset = atoi(argv[argi + 1]);
            argi += 2;
        } else if (strcmp(arg, "--action") == 0 && argi + 3 < argc) {
            render_cfg.use_preset = false;
            render_cfg.heading_action = parse_float_arg("heading_action", argv[argi + 1]);
            render_cfg.altitude_action = parse_float_arg("altitude_action", argv[argi + 2]);
            render_cfg.throttle_action = parse_float_arg("throttle_action", argv[argi + 3]);
            argi += 4;
        } else if (strcmp(arg, "--pos") == 0 && argi + 3 < argc) {
            render_cfg.pos_x = parse_float_arg("pos_x", argv[argi + 1]);
            render_cfg.pos_y = parse_float_arg("pos_y", argv[argi + 2]);
            render_cfg.pos_z = parse_float_arg("pos_z", argv[argi + 3]);
            argi += 4;
        } else if (strcmp(arg, "--pitch") == 0 && argi + 1 < argc) {
            render_cfg.pitch = parse_float_arg("pitch", argv[argi + 1]);
            argi += 2;
        } else if (strcmp(arg, "--speed") == 0 && argi + 1 < argc) {
            render_cfg.speed = parse_float_arg("speed", argv[argi + 1]);
            argi += 2;
        } else if (strcmp(arg, "--throttle-state") == 0 && argi + 1 < argc) {
            render_cfg.throttle_state = parse_float_arg("throttle_state", argv[argi + 1]);
            argi += 2;
        } else if (strcmp(arg, "--turn-seq") == 0) {
            render_cfg.use_turn_sequence = true;
            render_cfg.use_preset = false;
            render_cfg.altitude_action = -0.5f;
            render_cfg.throttle_action = 0.0f;
            argi += 1;
        } else if (strcmp(arg, "--turn-seq-period") == 0 && argi + 1 < argc) {
            render_cfg.sequence_period = atoi(argv[argi + 1]);
            argi += 2;
        } else {
            fprintf(stderr, "ERROR: Unknown or incomplete render argument: %s\n", arg);
            return 1;
        }
    }

    BvrEnv *env = calloc(1, sizeof(BvrEnv));
    env->num_agents = 1;
    env->framestack = 1;
    init(env);

    size_t obs_size = OBS_DIM * env->framestack;
    size_t act_size = 3;
    env->observations = (float *)calloc(env->num_agents * obs_size, sizeof(float));
    env->actions = (float *)calloc(env->num_agents * act_size, sizeof(float));
    env->rewards = (float *)calloc(env->num_agents, sizeof(float));
    env->terminals = (unsigned char *)calloc(env->num_agents, sizeof(unsigned char));

    if (!env->observations || !env->actions || !env->rewards) {
        fprintf(stderr, "ERROR: Failed to allocate memory for demo buffers.\n");
        free(env->observations);
        free(env->actions);
        free(env->rewards);
        free(env->terminals);
        free(env);
        return 0;
    }

    c_reset(env);
    apply_trim_config(env, &render_cfg);
    c_render(env);

    set_config_actions(env, &render_cfg);
    while (!WindowShouldClose()) {
        c_step(env);
        set_config_actions(env, &render_cfg);
        c_render(env);
    }

    c_close(env);
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free(env);

    return 0;
}
