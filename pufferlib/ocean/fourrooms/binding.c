#include "fourrooms.h"
#define Env Fourrooms
#include "../env_binding.h"


static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->size = (int)unpack(kwargs, "size");
    env->max_steps = (int)unpack(kwargs, "max_steps");
    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "targets_hit", log->on_targets);
    return 0;
}
