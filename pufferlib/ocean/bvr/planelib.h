// Inspried by Sam Turner and Finlay Sanders Drone Env (2025).

#pragma once

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

// Visualisation properties
#define WIDTH 1080
#define HEIGHT 720
#define TRAIL_LENGTH 50
#define HORIZON 1024

//THESE NEED ADAPTING FOR APPROPROATE SIZES FOR PLANES
//assumes ground at z=0

// World / simulation
#define GRID_X 150.0f
#define GRID_Y 150.0f
#define GRID_Z 150.0f
#define DT 0.05f
#define DT_RNG 0.0f

// Base aircraft physical parameters
#define BASE_MASS 8.0f
#define BASE_IXX 2.5f
#define BASE_IYY 3.5f
#define BASE_IZZ 5.0f
#define BASE_GRAVITY 9.81f

#define BASE_MAX_THRUST 120.0f
#define BASE_MAX_VEL 250.0f
#define BASE_MAX_OMEGA 8.0f
#define BASE_K_THROTTLE 1.0f

// Base geometry / atmosphere
#define BASE_WING_AREA 1.8f
#define BASE_WING_SPAN 3.0f
#define BASE_MEAN_CHORD 0.7f
#define BASE_RHO 1.225f

// Force coefficients
#define BASE_CL0 0.1f
#define BASE_CL_ALPHA 4.5f
#define BASE_CL_ELEVATOR 0.4f

#define BASE_CD0 0.03f
#define BASE_CD_ALPHA2 0.25f

#define BASE_CY_BETA -0.7f
#define BASE_CY_RUDDER 0.12f

// Moment coefficients
#define BASE_Cl_BETA -0.08f
#define BASE_Cl_AILERON 0.10f
#define BASE_Cl_P -0.45f
#define BASE_Cl_R 0.08f

#define BASE_Cm0 0.01f
#define BASE_Cm_ALPHA -1.2f
#define BASE_Cm_ELEVATOR -0.8f
#define BASE_Cm_Q -6.0f

#define BASE_Cn_BETA 0.18f
#define BASE_Cn_RUDDER -0.08f
#define BASE_Cn_P -0.03f
#define BASE_Cn_R -0.16f

// Controller gains
#define BASE_K_HEADING 1.5f
#define BASE_MAX_BANK_CMD 0.8f

#define BASE_K_BANK 2.0f
#define BASE_K_ROLL_RATE 0.4f

#define BASE_K_ALTITUDE 0.1f
#define BASE_MAX_PITCH_CMD 0.35f

#define BASE_K_PITCH 2.0f
#define BASE_K_PITCH_RATE 0.5f

#define BASE_K_BETA 0.5f
#define BASE_K_YAW_RATE 0.2f

// Initial state
#define BASE_INIT_X 0.0f
#define BASE_INIT_Y 0.0f
#define BASE_INIT_Z 100.0f

#define BASE_INIT_U 20.0f
#define BASE_INIT_V 0.0f
#define BASE_INIT_W 0.0f

#define BASE_INIT_P 0.0f
#define BASE_INIT_Q 0.0f
#define BASE_INIT_R 0.0f

#define BASE_INIT_THROTTLE 0.6f

// Corner to corner distance
#define MAX_DIST sqrtf((2*GRID_X)*(2*GRID_X) + (2*GRID_Y)*(2*GRID_Y) + (2*GRID_Z)*(2*GRID_Z))

typedef struct {
    float w, x, y, z;
} Quat;

typedef struct {
    float x, y, z;
} Vec3;

typedef struct {
    Vec3 pos[TRAIL_LENGTH];
    int index;
    int count;
} Trail;

typedef struct {
    Vec3 pos;      // global position (x, y, z)
    Vec3 vel;      // linear velocity (u, v, w)
    Quat quat;     // roll/pitch/yaw (phi/theta/psi) as a quaternion
    Vec3 omega;    // angular velocity (p, q, r)
    float throttle_state; // engine lag state
} State;

typedef struct {
    Vec3 vel;       // Derivative of position (pos_dot efectively)
    Vec3 v_dot;       // Derivative of velocity (linear acceleration)
    Quat q_dot;       // Derivative of quaternion (rate of change of orientation)
    Vec3 w_dot;       // Derivative of angular velocity (angular acceleration)
    float throttle_dot; // Derivative of throttle state (rate of change of throttle)
} StateDerivative;

typedef struct {
    float mass; // kg
    float ixx, iyy, izz; // kgm^2
    float gravity; // m/s^2
    
    float max_thrust; // N
    float max_vel; // m/s
    float max_omega; // rad/s
    float k_throttle; // throttle lag


    float wing_area;   // S
    float wing_span;   // b
    float mean_chord;  // c_bar
    float rho;         // air density
                          
    //force coefficients
    float CL0, CL_alpha, CL_elevator; //lift zero angle, lift angle of attack, elevator lift
    float CD0, CD_alpha2; //drag zero angle, drag angle of attack
    float CY_beta, CY_rudder; //side force from side slip, side force from rudder
                              
    //moment coefficients
    float Cl_beta, Cl_aileron, Cl_p, Cl_r; //roll from side slip, aileron roll, damping roll, yaw coupling
    float Cm0, Cm_alpha, Cm_elevator, Cm_q; //base pitch moment, stability, elevator authority, pitch rate damping
    float Cn_beta, Cn_rudder, Cn_p, Cn_r; //weathercock stability, rudder control, yaw damping, yaw coupling
} Params;

typedef struct {
    float throttle;  // [0, 1]
    float aileron;   // [-1, 1]
    float elevator;  // [-1, 1]
    float rudder;    // [-1, 1]
} Controls;

typedef struct {

    // k's are coefficients so if bank is proportional to heading error then we 
    // multiply by k_heading to get desired bank and same for other terms
    // heading -> bank
    float k_heading;
    float max_bank_cmd;   // rad

    // bank -> aileron
    float k_bank;
    float k_roll_rate;

    // altitude -> pitch
    float k_altitude;
    float max_pitch_cmd;  // rad

    // pitch -> elevator
    float k_pitch;
    float k_pitch_rate;

    // rudder stabilization
    float k_beta;
    float k_yaw_rate;
} ControlParams;

typedef struct {
    // core state and parameters
    State state;
    Params params;
    Vec3 prev_pos;

    // control parameters
    ControlParams control_params;

    // logging utils
    float episode_return;
    int episode_length;
    float score;
} Plane;

static inline float clampf(float v, float min, float max) {
    if (v < min)
        return min;
    if (v > max)
        return max;
    return v;
}

static inline float rndf(float a, float b) {
    return a + ((float)rand() / (float)RAND_MAX) * (b - a);
}

static inline Vec3 add3(Vec3 a, Vec3 b) { 
    return (Vec3){a.x + b.x, a.y + b.y, a.z + b.z}; 
}

static inline Quat add_quat(Quat a, Quat b) { 
    return (Quat){a.w + b.w, a.x + b.x, a.y + b.y, a.z + b.z}; 
}

static inline Vec3 sub3(Vec3 a, Vec3 b) { 
    return (Vec3){a.x - b.x, a.y - b.y, a.z - b.z}; 
}

static inline Vec3 scalmul3(Vec3 a, float b) { 
    return (Vec3){a.x * b, a.y * b, a.z * b}; 
}

static inline Quat scalmul_quat(Quat a, float b) { 
    return (Quat){a.w * b, a.x * b, a.y * b, a.z * b}; 
}

static inline float dot3(Vec3 a, Vec3 b) { 
    return a.x * b.x + a.y * b.y + a.z * b.z; 
}

static inline float norm3(Vec3 a) { 
    return sqrtf(dot3(a, a)); 
}

static inline void clamp3(Vec3 *vec, float min, float max) {
    vec->x = clampf(vec->x, min, max);
    vec->y = clampf(vec->y, min, max);
    vec->z = clampf(vec->z, min, max);
}

static inline Quat quat_mul(Quat q1, Quat q2) {
    Quat out;
    out.w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z;
    out.x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y;
    out.y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x;
    out.z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w;
    return out;
}

static inline void quat_normalize(Quat *q) {
    float n = sqrtf(q->w * q->w + q->x * q->x + q->y * q->y + q->z * q->z);
    if (n > 0.0f) {
        q->w /= n;
        q->x /= n;
        q->y /= n;
        q->z /= n;
    }
}

static inline Vec3 quat_rotate(Quat q, Vec3 v) {
    Quat qv = {0.0f, v.x, v.y, v.z};
    Quat tmp = quat_mul(q, qv);
    Quat q_conj = {q.w, -q.x, -q.y, -q.z};
    Quat res = quat_mul(tmp, q_conj);
    return (Vec3){res.x, res.y, res.z};
}

static inline Quat quat_inverse(Quat q) { 
    return (Quat){q.w, -q.x, -q.y, -q.z}; 
}

Quat rndquat() {
    float u1 = rndf(0.0f, 1.0f);
    float u2 = rndf(0.0f, 1.0f);
    float u3 = rndf(0.0f, 1.0f);

    float sqrt_1_minus_u1 = sqrtf(1.0f - u1);
    float sqrt_u1 = sqrtf(u1);

    float pi_2_u2 = 2.0f * M_PI * u2;
    float pi_2_u3 = 2.0f * M_PI * u3;

    Quat q;
    q.w = sqrt_1_minus_u1 * sinf(pi_2_u2);
    q.x = sqrt_1_minus_u1 * cosf(pi_2_u2);
    q.y = sqrt_u1 * sinf(pi_2_u3);
    q.z = sqrt_u1 * cosf(pi_2_u3);

    return q;
}

// force angle to be between -pi and pi
static inline float wrap_angle(float a) {
    while (a > (float)M_PI)  a -= 2.0f * (float)M_PI;
    while (a < -(float)M_PI) a += 2.0f * (float)M_PI;
    return a;
}

//convert quaterion to roll, pitch, yaw for use in control
static inline void quat_to_euler(Quat q, float *roll, float *pitch, float *yaw) {
    float sinr_cosp = 2.0f * (q.w * q.x + q.y * q.z);
    float cosr_cosp = 1.0f - 2.0f * (q.x * q.x + q.y * q.y);
    *roll = atan2f(sinr_cosp, cosr_cosp);

    float sinp = 2.0f * (q.w * q.y - q.z * q.x);
    if (fabsf(sinp) >= 1.0f) {
        *pitch = copysignf((float)M_PI / 2.0f, sinp);
    } else {
        *pitch = asinf(sinp);
    }

    float siny_cosp = 2.0f * (q.w * q.z + q.x * q.y);
    float cosy_cosp = 1.0f - 2.0f * (q.y * q.y + q.z * q.z);
    *yaw = atan2f(siny_cosp, cosy_cosp);
}

void action_to_controls(
    State *state,
    ControlParams *ctrl,
    float *actions,   // [desired_heading, desired_altitude, desired_throttle]
    Controls *out
) {
    // read actions and convert from [-1,1]
    float heading_cmd = wrap_angle(actions[0] * (float)M_PI);
    float altitude_cmd = 0.5f * (actions[1] + 1.0f) * GRID_Z;
    float throttle_cmd = clampf(0.5f * (actions[2] + 1.0f), 0.0f, 1.0f);

    //Comvert currnt state to roll pitch yaw
    float roll, pitch, yaw;
    quat_to_euler(state->quat, &roll, &pitch, &yaw);

    // Velocity in body f.o.r
    Vec3 v_body = quat_rotate(quat_inverse(state->quat), state->vel);
    float V = norm3(v_body);

    //beta is sideslip angle (0 = on nose, > 0 right, < 0 left)
    //ie how sideways the craft is moving relative to dir its pointing
    float beta = 0.0f;
    if (V > 1e-4f) {
        beta = atan2f(v_body.y, v_body.x);
    }

    // Heading target -> bank command
    // bank proportional to heading error so use bank error * bank coeff
    float heading_err = wrap_angle(heading_cmd - yaw);
    float bank_cmd = ctrl->k_heading * heading_err;
    bank_cmd = clampf(bank_cmd, -ctrl->max_bank_cmd, ctrl->max_bank_cmd);

    // Bank command -> aileron same logic as bank
    float bank_err = bank_cmd - roll;
    float aileron = ctrl->k_bank * bank_err
                  - ctrl->k_roll_rate * state->omega.x;
    aileron = clampf(aileron, -1.0f, 1.0f);

    // Altitude target -> pitch command same logic as bank
    float altitude_err = altitude_cmd - state->pos.z;
    float pitch_cmd = ctrl->k_altitude * altitude_err;
    pitch_cmd = clampf(pitch_cmd, -ctrl->max_pitch_cmd, ctrl->max_pitch_cmd);

    // Pitch command -> elevator
    float pitch_err = pitch_cmd - pitch;
    float elevator = ctrl->k_pitch * pitch_err
                   - ctrl->k_pitch_rate * state->omega.y;
    elevator = clampf(elevator, -1.0f, 1.0f);

    // Rudder for slip and yaw damping
    // if slding then apply rudder to damp slip
    float rudder = -ctrl->k_beta * beta
                 - ctrl->k_yaw_rate * state->omega.z;
    rudder = clampf(rudder, -1.0f, 1.0f);

    // Output commands
    out->throttle = throttle_cmd;
    out->aileron = aileron;
    out->elevator = elevator;
    out->rudder = rudder;
}

void init_plane(Plane* plane, float dr) {
    // physical parameters
    plane->params.mass = BASE_MASS * rndf(1.0f - dr, 1.0f + dr);
    plane->params.ixx = BASE_IXX * rndf(1.0f - dr, 1.0f + dr);
    plane->params.iyy = BASE_IYY * rndf(1.0f - dr, 1.0f + dr);
    plane->params.izz = BASE_IZZ * rndf(1.0f - dr, 1.0f + dr);
    plane->params.gravity = BASE_GRAVITY * rndf(0.99f, 1.01f);

    plane->params.max_thrust = BASE_MAX_THRUST * rndf(1.0f - dr, 1.0f + dr);
    plane->params.max_vel = BASE_MAX_VEL;
    plane->params.max_omega = BASE_MAX_OMEGA;
    plane->params.k_throttle = BASE_K_THROTTLE * rndf(1.0f - dr, 1.0f + dr);

    // geometry / atmosphere
    plane->params.wing_area = BASE_WING_AREA * rndf(1.0f - dr, 1.0f + dr);
    plane->params.wing_span = BASE_WING_SPAN * rndf(1.0f - dr, 1.0f + dr);
    plane->params.mean_chord = BASE_MEAN_CHORD * rndf(1.0f - dr, 1.0f + dr);
    plane->params.rho = BASE_RHO;

    // force coefficients
    plane->params.CL0 = BASE_CL0 * rndf(1.0f - dr, 1.0f + dr);
    plane->params.CL_alpha = BASE_CL_ALPHA * rndf(1.0f - dr, 1.0f + dr);
    plane->params.CL_elevator = BASE_CL_ELEVATOR * rndf(1.0f - dr, 1.0f + dr);

    plane->params.CD0 = BASE_CD0 * rndf(1.0f - dr, 1.0f + dr);
    plane->params.CD_alpha2 = BASE_CD_ALPHA2 * rndf(1.0f - dr, 1.0f + dr);

    plane->params.CY_beta = BASE_CY_BETA * rndf(1.0f - dr, 1.0f + dr);
    plane->params.CY_rudder = BASE_CY_RUDDER * rndf(1.0f - dr, 1.0f + dr);

    // moment coefficients
    plane->params.Cl_beta = BASE_Cl_BETA * rndf(1.0f - dr, 1.0f + dr);
    plane->params.Cl_aileron = BASE_Cl_AILERON * rndf(1.0f - dr, 1.0f + dr);
    plane->params.Cl_p = BASE_Cl_P * rndf(1.0f - dr, 1.0f + dr);
    plane->params.Cl_r = BASE_Cl_R * rndf(1.0f - dr, 1.0f + dr);

    plane->params.Cm0 = BASE_Cm0 * rndf(1.0f - dr, 1.0f + dr);
    plane->params.Cm_alpha = BASE_Cm_ALPHA * rndf(1.0f - dr, 1.0f + dr);
    plane->params.Cm_elevator = BASE_Cm_ELEVATOR * rndf(1.0f - dr, 1.0f + dr);
    plane->params.Cm_q = BASE_Cm_Q * rndf(1.0f - dr, 1.0f + dr);

    plane->params.Cn_beta = BASE_Cn_BETA * rndf(1.0f - dr, 1.0f + dr);
    plane->params.Cn_rudder = BASE_Cn_RUDDER * rndf(1.0f - dr, 1.0f + dr);
    plane->params.Cn_p = BASE_Cn_P * rndf(1.0f - dr, 1.0f + dr);
    plane->params.Cn_r = BASE_Cn_R * rndf(1.0f - dr, 1.0f + dr);

    // controller gains
    plane->control_params.k_heading = BASE_K_HEADING;
    plane->control_params.max_bank_cmd = BASE_MAX_BANK_CMD;

    plane->control_params.k_bank = BASE_K_BANK;
    plane->control_params.k_roll_rate = BASE_K_ROLL_RATE;

    plane->control_params.k_altitude = BASE_K_ALTITUDE;
    plane->control_params.max_pitch_cmd = BASE_MAX_PITCH_CMD;

    plane->control_params.k_pitch = BASE_K_PITCH;
    plane->control_params.k_pitch_rate = BASE_K_PITCH_RATE;

    plane->control_params.k_beta = BASE_K_BETA;
    plane->control_params.k_yaw_rate = BASE_K_YAW_RATE;

    // initial state
    plane->state.pos = (Vec3){BASE_INIT_X, BASE_INIT_Y, BASE_INIT_Z};
    plane->state.vel = (Vec3){BASE_INIT_U, BASE_INIT_V, BASE_INIT_W};
    plane->state.quat = (Quat){1.0f, 0.0f, 0.0f, 0.0f};
    plane->state.omega = (Vec3){BASE_INIT_P, BASE_INIT_Q, BASE_INIT_R};
    plane->state.throttle_state = BASE_INIT_THROTTLE;

    plane->prev_pos = plane->state.pos;

    // logging
    plane->episode_return = 0.0f;
    plane->episode_length = 0;
    plane->score = 0.0f;
}

void compute_derivatives(State* state, Params* params, Controls* cmd, StateDerivative* derivatives){
    //this takes in the CONTROLS not the raw actions since actions 
    //actions are desired heading, desired altitude, desired throttle
    //cmd contains throttle, aileron, elevator, rudder

    float throttle_dot = (cmd->throttle - state->throttle_state) / params->k_throttle;

    //thrust
    float T;
    T = params->max_thrust * state->throttle_state;

    //thrust force
    Vec3 F_thrust_body= {T,0.0f, 0.0f};

    //state->vel is the velocity of the body in world frame of reference
    //v_body is the velocity of the body in aircraft frame of reference
    Vec3 v_body = quat_rotate(quat_inverse(state->quat), state->vel);

    //q is dynamic pressure
    float V = norm3(v_body);
    float q = 0.5f * params->rho * V * V;

    float alpha = 0.0f;  // angle of attack
    float beta = 0.0f; // sideslip
    if (V > 1e-4f) {
        alpha = atan2f(v_body.z, v_body.x);
        beta = atan2f(v_body.y, v_body.x);
    }

                                               
    float CL = params->CL0 
         + params->CL_alpha * alpha 
         + params->CL_elevator * cmd->elevator;

    float CD = params->CD0 
         + params->CD_alpha2 * alpha * alpha;

    float CY = params->CY_beta * beta 
         + params->CY_rudder * cmd->rudder;

    //lift force
    float L = q * params->wing_area * CL; //lift
    float D = q * params->wing_area * CD; //drag
    float Yf = q * params->wing_area * CY; //side force
    
    // aero force
    Vec3 F_aero_body = {-D, Yf, -L};

    //Total force
    Vec3 F_body = add3(F_thrust_body, F_aero_body);

    //rotate into world frame
    Vec3 F_world = quat_rotate(state->quat, F_body);

    //roll pitch yaw moments

    float Cl = params->Cl_beta * beta
         + params->Cl_aileron * cmd->aileron
         + params->Cl_p * state->omega.x
         + params->Cl_r * state->omega.z;

    float Cm = params->Cm0
             + params->Cm_alpha * alpha
             + params->Cm_elevator * cmd->elevator
             + params->Cm_q * state->omega.y;

    float Cn = params->Cn_beta * beta
             + params->Cn_rudder * cmd->rudder
             + params->Cn_p * state->omega.x
             + params->Cn_r * state->omega.z;

    Vec3 Tau;
    Tau.x = q * params->wing_area * params->wing_span * Cl;
    Tau.y = q * params->wing_area * params->mean_chord * Cm;
    Tau.z = q * params->wing_area * params->wing_span * Cn;
    
    // velocity rates, a = F/m
    Vec3 v_dot;
    v_dot.x = F_world.x / params->mass;
    v_dot.y = F_world.y / params->mass;
    v_dot.z = (F_world.z/ params->mass) - params->gravity;

    // quaternion rates
    Quat omega_q = {0.0f, state->omega.x, state->omega.y, state->omega.z};
    Quat q_dot = quat_mul(state->quat, omega_q);
    q_dot.w *= 0.5f;
    q_dot.x *= 0.5f;
    q_dot.y *= 0.5f;
    q_dot.z *= 0.5f;

    // gyroscopic torque
    Vec3 Tau_iner;
    Tau_iner.x = (params->iyy - params->izz) * state->omega.y * state->omega.z;
    Tau_iner.y = (params->izz - params->ixx) * state->omega.z * state->omega.x;
    Tau_iner.z = (params->ixx - params->iyy) * state->omega.x * state->omega.y;

    // angular velocity rates
    Vec3 w_dot;
    w_dot.x = (Tau.x + Tau_iner.x) / params->ixx;
    w_dot.y = (Tau.y + Tau_iner.y) / params->iyy;
    w_dot.z = (Tau.z + Tau_iner.z) / params->izz;

    derivatives->vel = state->vel;
    derivatives->v_dot = v_dot;
    derivatives->q_dot = q_dot;
    derivatives->w_dot = w_dot;
    derivatives->throttle_dot = throttle_dot;
}

static inline void integrate_state(
    State* initial, StateDerivative* deriv, float dt, State* output) {
    output->pos = add3(initial->pos, scalmul3(deriv->vel, dt));
    output->vel = add3(initial->vel, scalmul3(deriv->v_dot, dt));
    output->quat = add_quat(initial->quat, scalmul_quat(deriv->q_dot, dt));
    output->omega = add3(initial->omega, scalmul3(deriv->w_dot, dt));
    output->throttle_state = initial->throttle_state + deriv->throttle_dot * dt;
    quat_normalize(&output->quat);
}

void rk4_step(State* state, Params* params, ControlParams* ctrl, float* actions, float dt) {
    StateDerivative k1, k2, k3, k4;
    State temp_state;
    Controls cmd;

    action_to_controls(state, ctrl, actions, &cmd);
    compute_derivatives(state, params, &cmd, &k1);

    integrate_state(state, &k1, dt * 0.5f, &temp_state);
    action_to_controls(&temp_state, ctrl, actions, &cmd);
    compute_derivatives(&temp_state, params, &cmd, &k2);

    integrate_state(state, &k2, dt * 0.5f, &temp_state);
    action_to_controls(&temp_state, ctrl, actions, &cmd);
    compute_derivatives(&temp_state, params, &cmd, &k3);

    integrate_state(state, &k3, dt, &temp_state);
    action_to_controls(&temp_state, ctrl, actions, &cmd);
    compute_derivatives(&temp_state, params, &cmd, &k4);

    float dt_6 = dt / 6.0f;

    state->pos.x += (k1.vel.x + 2.0f * k2.vel.x + 2.0f * k3.vel.x + k4.vel.x) * dt_6;
    state->pos.y += (k1.vel.y + 2.0f * k2.vel.y + 2.0f * k3.vel.y + k4.vel.y) * dt_6;
    state->pos.z += (k1.vel.z + 2.0f * k2.vel.z + 2.0f * k3.vel.z + k4.vel.z) * dt_6;

    state->vel.x += (k1.v_dot.x + 2.0f * k2.v_dot.x + 2.0f * k3.v_dot.x + k4.v_dot.x) * dt_6;
    state->vel.y += (k1.v_dot.y + 2.0f * k2.v_dot.y + 2.0f * k3.v_dot.y + k4.v_dot.y) * dt_6;
    state->vel.z += (k1.v_dot.z + 2.0f * k2.v_dot.z + 2.0f * k3.v_dot.z + k4.v_dot.z) * dt_6;

    state->quat.w += (k1.q_dot.w + 2.0f * k2.q_dot.w + 2.0f * k3.q_dot.w + k4.q_dot.w) * dt_6;
    state->quat.x += (k1.q_dot.x + 2.0f * k2.q_dot.x + 2.0f * k3.q_dot.x + k4.q_dot.x) * dt_6;
    state->quat.y += (k1.q_dot.y + 2.0f * k2.q_dot.y + 2.0f * k3.q_dot.y + k4.q_dot.y) * dt_6;
    state->quat.z += (k1.q_dot.z + 2.0f * k2.q_dot.z + 2.0f * k3.q_dot.z + k4.q_dot.z) * dt_6;

    state->omega.x += (k1.w_dot.x + 2.0f * k2.w_dot.x + 2.0f * k3.w_dot.x + k4.w_dot.x) * dt_6;
    state->omega.y += (k1.w_dot.y + 2.0f * k2.w_dot.y + 2.0f * k3.w_dot.y + k4.w_dot.y) * dt_6;
    state->omega.z += (k1.w_dot.z + 2.0f * k2.w_dot.z + 2.0f * k3.w_dot.z + k4.w_dot.z) * dt_6;

    state->throttle_state += (k1.throttle_dot + 2.0f * k2.throttle_dot + 2.0f * k3.throttle_dot + k4.throttle_dot) * dt_6;

    quat_normalize(&state->quat);
}

void move_aircraft(Plane* plane, float* actions) {
    // clamp high-level actions though shouldnt be needed
    actions[0] = clampf(actions[0], -1.0f, 1.0f); // heading target representation
    actions[1] = clampf(actions[1], -1.0f, 1.0f); // altitude target representation
    actions[2] = clampf(actions[2],  -1.0f, 1.0f); // throttle target, if already [0,1]

    // domain randomized dt
    float dt = DT * rndf(1.0f - DT_RNG, 1.0f + DT_RNG);

    // save previous position
    plane->prev_pos = plane->state.pos;

    // update aircraft state
    rk4_step(&plane->state, &plane->params, &plane->control_params, actions, dt);

    // clamp for stability / observations
    clamp3(&plane->state.vel, -plane->params.max_vel, plane->params.max_vel);
    clamp3(&plane->state.omega, -plane->params.max_omega, plane->params.max_omega);
    plane->state.throttle_state = clampf(plane->state.throttle_state, 0.0f, 1.0f);
}
