#pragma once

#include "planelib.h"

#define MISSILE_DT 0.02f

#define MISSILE_BASE_MASS 12.0f
#define MISSILE_BASE_IXX 0.15f
#define MISSILE_BASE_IYY 1.20f
#define MISSILE_BASE_IZZ 1.20f
#define MISSILE_BASE_GRAVITY 9.81f

#define MISSILE_BASE_REF_AREA 0.03f
#define MISSILE_BASE_REF_LENGTH 2.5f
#define MISSILE_BASE_RHO 1.225f

#define MISSILE_BASE_THRUST_BOOST 420.0f
#define MISSILE_BASE_THRUST_SUSTAIN 260.0f
#define MISSILE_BASE_BOOST_TIME 1.5f
#define MISSILE_BASE_SUSTAIN_TIME 4.0f

#define MISSILE_BASE_CD0 0.20f
#define MISSILE_BASE_CD_ALPHA2 1.20f

#define MISSILE_BASE_CY_BETA -2.0f
#define MISSILE_BASE_CZ_ALPHA -6.0f

#define MISSILE_BASE_CM_ALPHA -1.5f
#define MISSILE_BASE_CM_Q -8.0f
#define MISSILE_BASE_CM_FIN -8.0f

#define MISSILE_BASE_CN_BETA 1.5f
#define MISSILE_BASE_CN_R -6.0f
#define MISSILE_BASE_CN_FIN -8.0f

#define MISSILE_BASE_MAX_VEL 260.0f
#define MISSILE_BASE_MAX_OMEGA 28.0f

#define MISSILE_GUIDANCE_K_YAW 1.0f
#define MISSILE_GUIDANCE_K_PITCH 0.8f
#define MISSILE_GUIDANCE_K_R 1.2f
#define MISSILE_GUIDANCE_K_Q 1.8f
#define MISSILE_PN_GAIN 3.0f

typedef struct {
    Vec3 pos;
    Vec3 vel;
    Quat quat;
    Vec3 omega;
    float motor_time;
} MissileState;

typedef struct {
    Vec3 vel;
    Vec3 v_dot;
    Quat q_dot;
    Vec3 w_dot;
    float motor_time_dot;
} MissileStateDerivative;

typedef struct {
    float mass;
    float ixx, iyy, izz;
    float gravity;

    float ref_area;
    float ref_length;
    float rho;

    float thrust_boost;
    float thrust_sustain;
    float boost_time;
    float sustain_time;

    float CD0;
    float CD_alpha2;

    float CY_beta;
    float CZ_alpha;

    float Cm_alpha;
    float Cm_q;
    float Cm_fin;

    float Cn_beta;
    float Cn_r;
    float Cn_fin;

    float max_vel;
    float max_omega;
} MissileParams;

typedef struct {
    float fin_pitch;
    float fin_yaw;
} MissileControls;

typedef struct {
    MissileState state;
    MissileParams params;
} Missile;

static inline Quat missile_quat_from_yaw_pitch(float yaw, float pitch) {
    float cy = cosf(0.5f * yaw);
    float sy = sinf(0.5f * yaw);
    float cp = cosf(0.5f * pitch);
    float sp = sinf(0.5f * pitch);

    Quat q = {
        .w = cy * cp,
        .x = -sy * sp,
        .y = cy * sp,
        .z = sy * cp,
    };
    quat_normalize(&q);
    return q;
}

static inline void aim_missile_at_point(Missile *m, Vec3 target_pos, float launch_speed) {
    Vec3 los = sub3(target_pos, m->state.pos);
    float d = norm3(los);
    if (d <= 1e-4f) {
        return;
    }

    Vec3 dir = scalmul3(los, 1.0f / d);
    float yaw = atan2f(dir.y, dir.x);
    float pitch = atan2f(-dir.z, sqrtf(dir.x * dir.x + dir.y * dir.y));

    m->state.vel = scalmul3(dir, launch_speed);
    m->state.quat = missile_quat_from_yaw_pitch(yaw, pitch);
}

static inline float missile_thrust(float t, const MissileParams *p) {
    if (t < p->boost_time) return p->thrust_boost;
    if (t < p->boost_time + p->sustain_time) return p->thrust_sustain;
    return 0.0f;
}

static inline void target_to_missile_controls(
    const MissileState *state,
    Vec3 target_pos,
    Vec3 target_vel,
    MissileControls *cmd
) {
    Vec3 rel_pos = sub3(target_pos, state->pos);
    Vec3 rel_vel = sub3(target_vel, state->vel);
    Vec3 los_body = quat_rotate(quat_inverse(state->quat), rel_pos);

    float yaw_err = atan2f(los_body.y, los_body.x);
    float pitch_err = atan2f(-los_body.z, los_body.x);

    float range_xy2 = rel_pos.x * rel_pos.x + rel_pos.y * rel_pos.y;
    float range_xy = sqrtf(range_xy2);
    float range2 = range_xy2 + rel_pos.z * rel_pos.z;
    float yaw_rate_los = 0.0f;
    float pitch_rate_los = 0.0f;
    float rho_dot = 0.0f;
    float closing_speed = 0.0f;

    if (range_xy2 > 1e-6f) {
        yaw_rate_los = (rel_pos.x * rel_vel.y - rel_pos.y * rel_vel.x) / range_xy2;
        rho_dot = (rel_pos.x * rel_vel.x + rel_pos.y * rel_vel.y) / range_xy;
    }
    if (range2 > 1e-6f) {
        pitch_rate_los = ((-rel_vel.z) * range_xy + rel_pos.z * rho_dot) / range2;
        closing_speed = -dot3(rel_pos, rel_vel) / sqrtf(range2);
    }

    float pn_yaw = MISSILE_PN_GAIN * closing_speed * yaw_rate_los;
    float pn_pitch = MISSILE_PN_GAIN * closing_speed * pitch_rate_los;

    cmd->fin_yaw = clampf(
        MISSILE_GUIDANCE_K_YAW * yaw_err + pn_yaw
        - MISSILE_GUIDANCE_K_R * state->omega.z,
        -1.0f, 1.0f
    );

    cmd->fin_pitch = clampf(
        MISSILE_GUIDANCE_K_PITCH * pitch_err + pn_pitch
        - MISSILE_GUIDANCE_K_Q * state->omega.y,
        -1.0f, 1.0f
    );
}

static inline void init_missile(Missile *m) {
    m->params.mass = MISSILE_BASE_MASS;
    m->params.ixx = MISSILE_BASE_IXX;
    m->params.iyy = MISSILE_BASE_IYY;
    m->params.izz = MISSILE_BASE_IZZ;
    m->params.gravity = MISSILE_BASE_GRAVITY;

    m->params.ref_area = MISSILE_BASE_REF_AREA;
    m->params.ref_length = MISSILE_BASE_REF_LENGTH;
    m->params.rho = MISSILE_BASE_RHO;

    m->params.thrust_boost = MISSILE_BASE_THRUST_BOOST;
    m->params.thrust_sustain = MISSILE_BASE_THRUST_SUSTAIN;
    m->params.boost_time = MISSILE_BASE_BOOST_TIME;
    m->params.sustain_time = MISSILE_BASE_SUSTAIN_TIME;

    m->params.CD0 = MISSILE_BASE_CD0;
    m->params.CD_alpha2 = MISSILE_BASE_CD_ALPHA2;

    m->params.CY_beta = MISSILE_BASE_CY_BETA;
    m->params.CZ_alpha = MISSILE_BASE_CZ_ALPHA;

    m->params.Cm_alpha = MISSILE_BASE_CM_ALPHA;
    m->params.Cm_q = MISSILE_BASE_CM_Q;
    m->params.Cm_fin = MISSILE_BASE_CM_FIN;

    m->params.Cn_beta = MISSILE_BASE_CN_BETA;
    m->params.Cn_r = MISSILE_BASE_CN_R;
    m->params.Cn_fin = MISSILE_BASE_CN_FIN;

    m->params.max_vel = MISSILE_BASE_MAX_VEL;
    m->params.max_omega = MISSILE_BASE_MAX_OMEGA;

    m->state.pos = (Vec3){0.0f, 0.0f, 50.0f};
    m->state.vel = (Vec3){150.0f, 0.0f, 0.0f};
    m->state.quat = (Quat){1.0f, 0.0f, 0.0f, 0.0f};
    m->state.omega = (Vec3){0.0f, 0.0f, 0.0f};
    m->state.motor_time = 0.0f;
}

static inline void compute_missile_derivatives(
    const MissileState *state,
    const MissileParams *params,
    const MissileControls *cmd,
    MissileStateDerivative *deriv
) {
    Vec3 v_body = quat_rotate(quat_inverse(state->quat), state->vel);
    float V = norm3(v_body);
    float qbar = 0.5f * params->rho * V * V;

    float alpha = 0.0f;
    float beta = 0.0f;
    if (V > 1e-4f) {
        alpha = atan2f(v_body.z, v_body.x);
        beta = atan2f(v_body.y, v_body.x);
    }

    float thrust = missile_thrust(state->motor_time, params);

    float CD = params->CD0 + params->CD_alpha2 * alpha * alpha;
    float D = qbar * params->ref_area * CD;

    float Y = qbar * params->ref_area * (params->CY_beta * beta + cmd->fin_yaw);
    float Z = qbar * params->ref_area * (params->CZ_alpha * alpha + cmd->fin_pitch);

    Vec3 F_thrust_body = {thrust, 0.0f, 0.0f};
    Vec3 F_aero_body = {-D, Y, Z};
    Vec3 F_body = add3(F_thrust_body, F_aero_body);

    Vec3 F_world = quat_rotate(state->quat, F_body);

    Vec3 v_dot;
    v_dot.x = F_world.x / params->mass;
    v_dot.y = F_world.y / params->mass;
    v_dot.z = (F_world.z / params->mass) - params->gravity;

    Quat omega_q = {0.0f, state->omega.x, state->omega.y, state->omega.z};
    Quat q_dot = quat_mul(state->quat, omega_q);
    q_dot.w *= 0.5f;
    q_dot.x *= 0.5f;
    q_dot.y *= 0.5f;
    q_dot.z *= 0.5f;

    Vec3 Tau;
    Tau.x = 0.0f;
    Tau.y = qbar * params->ref_area * params->ref_length *
            (params->Cm_alpha * alpha + params->Cm_q * state->omega.y + params->Cm_fin * cmd->fin_pitch);
    Tau.z = qbar * params->ref_area * params->ref_length *
            (params->Cn_beta * beta + params->Cn_r * state->omega.z + params->Cn_fin * cmd->fin_yaw);

    Vec3 Tau_iner;
    Tau_iner.x = (params->iyy - params->izz) * state->omega.y * state->omega.z;
    Tau_iner.y = (params->izz - params->ixx) * state->omega.z * state->omega.x;
    Tau_iner.z = (params->ixx - params->iyy) * state->omega.x * state->omega.y;

    Vec3 w_dot;
    w_dot.x = (Tau.x + Tau_iner.x) / params->ixx;
    w_dot.y = (Tau.y + Tau_iner.y) / params->iyy;
    w_dot.z = (Tau.z + Tau_iner.z) / params->izz;

    deriv->vel = state->vel;
    deriv->v_dot = v_dot;
    deriv->q_dot = q_dot;
    deriv->w_dot = w_dot;
    deriv->motor_time_dot = 1.0f;
}

static inline void missile_step(
    const MissileState *initial,
    const MissileStateDerivative *deriv,
    float dt,
    MissileState *out
) {
    out->pos = add3(initial->pos, scalmul3(deriv->vel, dt));
    out->vel = add3(initial->vel, scalmul3(deriv->v_dot, dt));
    out->quat = add_quat(initial->quat, scalmul_quat(deriv->q_dot, dt));
    out->omega = add3(initial->omega, scalmul3(deriv->w_dot, dt));
    out->motor_time = initial->motor_time + deriv->motor_time_dot * dt;
    quat_normalize(&out->quat);
}

static inline void missile_rk4_step(
    MissileState *state,
    const MissileParams *params,
    Vec3 target_pos,
    Vec3 target_vel,
    float dt
) {
    MissileStateDerivative k1, k2, k3, k4;
    MissileState temp_state;
    MissileControls cmd;

    target_to_missile_controls(state, target_pos, target_vel, &cmd);
    compute_missile_derivatives(state, params, &cmd, &k1);

    missile_step(state, &k1, dt * 0.5f, &temp_state);
    target_to_missile_controls(&temp_state, target_pos, target_vel, &cmd);
    compute_missile_derivatives(&temp_state, params, &cmd, &k2);

    missile_step(state, &k2, dt * 0.5f, &temp_state);
    target_to_missile_controls(&temp_state, target_pos, target_vel, &cmd);
    compute_missile_derivatives(&temp_state, params, &cmd, &k3);

    missile_step(state, &k3, dt, &temp_state);
    target_to_missile_controls(&temp_state, target_pos, target_vel, &cmd);
    compute_missile_derivatives(&temp_state, params, &cmd, &k4);

    float dt_6 = dt / 6.0f;

    state->pos.x += (k1.vel.x + 2.0f*k2.vel.x + 2.0f*k3.vel.x + k4.vel.x) * dt_6;
    state->pos.y += (k1.vel.y + 2.0f*k2.vel.y + 2.0f*k3.vel.y + k4.vel.y) * dt_6;
    state->pos.z += (k1.vel.z + 2.0f*k2.vel.z + 2.0f*k3.vel.z + k4.vel.z) * dt_6;

    state->vel.x += (k1.v_dot.x + 2.0f*k2.v_dot.x + 2.0f*k3.v_dot.x + k4.v_dot.x) * dt_6;
    state->vel.y += (k1.v_dot.y + 2.0f*k2.v_dot.y + 2.0f*k3.v_dot.y + k4.v_dot.y) * dt_6;
    state->vel.z += (k1.v_dot.z + 2.0f*k2.v_dot.z + 2.0f*k3.v_dot.z + k4.v_dot.z) * dt_6;

    state->quat.w += (k1.q_dot.w + 2.0f*k2.q_dot.w + 2.0f*k3.q_dot.w + k4.q_dot.w) * dt_6;
    state->quat.x += (k1.q_dot.x + 2.0f*k2.q_dot.x + 2.0f*k3.q_dot.x + k4.q_dot.x) * dt_6;
    state->quat.y += (k1.q_dot.y + 2.0f*k2.q_dot.y + 2.0f*k3.q_dot.y + k4.q_dot.y) * dt_6;
    state->quat.z += (k1.q_dot.z + 2.0f*k2.q_dot.z + 2.0f*k3.q_dot.z + k4.q_dot.z) * dt_6;

    state->omega.x += (k1.w_dot.x + 2.0f*k2.w_dot.x + 2.0f*k3.w_dot.x + k4.w_dot.x) * dt_6;
    state->omega.y += (k1.w_dot.y + 2.0f*k2.w_dot.y + 2.0f*k3.w_dot.y + k4.w_dot.y) * dt_6;
    state->omega.z += (k1.w_dot.z + 2.0f*k2.w_dot.z + 2.0f*k3.w_dot.z + k4.w_dot.z) * dt_6;

    state->motor_time += (k1.motor_time_dot + 2.0f*k2.motor_time_dot + 2.0f*k3.motor_time_dot + k4.motor_time_dot) * dt_6;

    quat_normalize(&state->quat);
}

static inline void move_missile(Missile *m, Vec3 target_pos, Vec3 target_vel) {
    missile_rk4_step(&m->state, &m->params, target_pos, target_vel, MISSILE_DT);
    clamp3(&m->state.vel, -m->params.max_vel, m->params.max_vel);
    clamp3(&m->state.omega, -m->params.max_omega, m->params.max_omega);
}
