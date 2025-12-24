#include "hanoi.h"

#define Env Hanoi
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->pegs = unpack(kwargs, "pegs");
    env->disks = unpack(kwargs, "disks");
    env->max_timesteps = unpack(kwargs, "max_timesteps");
    env->celebrate_ticks = unpack(kwargs, "celebrate_ticks");
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    return 0;
}
