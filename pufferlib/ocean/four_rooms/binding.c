#include "four_rooms.h"
#include <Python.h> 

static PyObject* my_vec_get(PyObject* self, PyObject* args);
#define MY_METHODS {"my_vec_get", my_vec_get, METH_VARARGS, "Get positions from all envs"}

#define Env FourRooms
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->size = unpack(kwargs, "size");
    env->see_through_walls = 0;
    // Allocate grid memory for full state (stores OBJECT_IDX values)
    env->grid = (unsigned char*)calloc(env->size * env->size, sizeof(unsigned char));
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    return 0;
}


static PyObject* my_vec_get(PyObject* self, PyObject* args) {
    VecEnv* vec = unpack_vecenv(args);
    if (!vec) return NULL;

    PyObject* dict = PyDict_New();

    for (int e = 0; e < vec->num_envs; e++) {
        Env* env = vec->envs[e];

        // Make a sub-dict for this env
        PyObject* d = PyDict_New();

        // agent position
        PyObject* pos = Py_BuildValue("(dd)", (double)env->agent_x, (double)env->agent_y);
        PyDict_SetItemString(d, "pos", pos);
        Py_DECREF(pos);

        // wrap env->grid as a NumPy array
        npy_intp dims[2] = {env->size, env->size};
        PyObject* grid_np = PyArray_SimpleNewFromData(
            2, dims, NPY_UINT8, (void*)env->grid);
        PyArray_CLEARFLAGS((PyArrayObject*)grid_np, NPY_ARRAY_WRITEABLE);
        PyDict_SetItemString(d, "grid", grid_np);
        Py_DECREF(grid_np);

        // key for outer dict: "env0", "env1", ...
        char key[32];
        snprintf(key, sizeof(key), "env%d", e);
        PyDict_SetItemString(dict, key, d);
        Py_DECREF(d);
    }

    return dict;
}
