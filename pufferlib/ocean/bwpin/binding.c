#include "bwpin.h"
#include <Python.h>

static PyObject* vec_goal_set(PyObject* self, PyObject* args);
static PyObject* vec_shuffle_moves(PyObject* self, PyObject* args);
static PyObject* vec_shuffle_all(PyObject* self, PyObject* args);

#define Env Hanoi

#define MY_METHODS \
      {"vec_goal_set", vec_goal_set, METH_VARARGS, "Set env_id to the solved goal state"}, \
      {"vec_shuffle_moves", vec_shuffle_moves, METH_VARARGS, "Apply n valid reverse moves starting from the goal state"}, \
      {"vec_shuffle_all", vec_shuffle_all, METH_VARARGS, "Apply n valid reverse moves starting from all goal states"}
      

#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->pegs = unpack(kwargs, "pegs");
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

static PyObject* vec_goal_set(PyObject* self, PyObject* args) {
      if (PyTuple_Size(args) != 2) {
          PyErr_SetString(PyExc_TypeError, "vec_goal_set(vec_handle, env_id)");
          return NULL;
      }
      VecEnv* vec = unpack_vecenv(args);
      if (!vec) {
          return NULL;
      }
      PyObject* env_id_obj = PyTuple_GetItem(args, 1);
      if (!PyLong_Check(env_id_obj)) {
          PyErr_SetString(PyExc_TypeError, "env_id must be an integer");
          return NULL;
      }
      long env_id = PyLong_AsLong(env_id_obj);
      if (env_id < 0 || env_id >= vec->num_envs) {
          PyErr_SetString(PyExc_ValueError, "env_id out of range");
          return NULL;
      }
      goal_set((Hanoi*)vec->envs[env_id]);
      Py_RETURN_NONE;
  }

static PyObject* vec_shuffle_moves(PyObject* self, PyObject* args) {
    if (PyTuple_Size(args) != 3) {
        PyErr_SetString(PyExc_TypeError, "vec_shuffle_moves(vec_handle, env_id, num_moves)");
        return NULL;
    }
    VecEnv* vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }
    PyObject* env_id_obj = PyTuple_GetItem(args, 1);
    PyObject* moves_obj = PyTuple_GetItem(args, 2);
    if (!PyLong_Check(env_id_obj) || !PyLong_Check(moves_obj)) {
        PyErr_SetString(PyExc_TypeError, "env_id and num_moves must be integers");
        return NULL;
    }
    long env_id = PyLong_AsLong(env_id_obj);
    long moves = PyLong_AsLong(moves_obj);
    if (env_id < 0 || env_id >= vec->num_envs) {
        PyErr_SetString(PyExc_ValueError, "env_id out of range");
        return NULL;
    }
    if (moves < 0) {
        PyErr_SetString(PyExc_ValueError, "num_moves must be >= 0");
        return NULL;
    }
    shuffle_moves((Hanoi*)vec->envs[env_id], (int)moves);
    Py_RETURN_NONE;
}

static PyObject* vec_shuffle_all(PyObject* self, PyObject* args) {
         if (PyTuple_Size(args) != 2) {
             PyErr_SetString(PyExc_TypeError, "vec_shuffle_all(vec_handle, num_moves)");
             return NULL;
         }
         VecEnv* vec = unpack_vecenv(args);
         if (!vec) {
             return NULL;
         }
         PyObject* moves_obj = PyTuple_GetItem(args, 1);
         if (!PyLong_Check(moves_obj)) {
             PyErr_SetString(PyExc_TypeError, "num_moves must be an integer");
             return NULL;
         }
         long moves = PyLong_AsLong(moves_obj);
         if (moves < 0) {
             PyErr_SetString(PyExc_ValueError, "num_moves must be >= 0");
             return NULL;
         }
         for (int i = 0; i < vec->num_envs; ++i) {
             shuffle_moves((Hanoi*)vec->envs[i], (int)moves);
         }
         Py_RETURN_NONE;
     }


