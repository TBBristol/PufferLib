#include "stacking.h"

#define Env ContainerStacking

#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->max_height = unpack(kwargs, "max_height");
    env->num_stacks =  unpack(kwargs, "num_stacks");
    env->num_containers = unpack(kwargs, "num_containers");
    env->max_ep_steps    = unpack(kwargs, "max_ep_steps");
    env->reward_us = unpack(kwargs, "reward_us");
    env->reward_max_breach = unpack(kwargs, "reward_max_breach");
    env->reset_max_breach = unpack(kwargs, "reset_max_breach");
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "num_invalids", log->num_invalids);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "n", log->n);
    return 0;
}