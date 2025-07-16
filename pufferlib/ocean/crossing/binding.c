#include "crossing.h"

#define Env RiverCrossing
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->boats           = unpack(kwargs, "boats");
    env->passengers      = unpack(kwargs, "passengers");
    env->max_passengers  = unpack(kwargs, "max_passengers");
    env->max_boats       = unpack(kwargs, "max_boats");
    env->max_ep_steps    = unpack(kwargs, "max_ep_steps");
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    return 0;
}