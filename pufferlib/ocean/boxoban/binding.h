#define BOXOBAN_MAPS_IMPLEMENTATION //enables mmap
#include "boxoban.h"

#define OBS_SIZE 400
#define NUM_ATNS 1
#define ACT_SIZES {5}
#define OBS_TYPE UNSIGNED_CHAR
#define ACT_TYPE DOUBLE

#define Env Boxoban
#include "env_binding.h"

static int parse_difficulty_id(Dict* kwargs) {
  int difficulty_id = 0;
  DictItem* difficulty_item = dict_get_unsafe(kwargs, "difficulty");
  if (difficulty_item == NULL) {
    return difficulty_id;
  }

  double raw = difficulty_item->value;
  int parsed = (int)raw;
  if (raw != (double)parsed) {
    fprintf(stderr,
      "Boxoban 'difficulty' must be an integer in [0, 4], got %.6f. Falling back to 0 (basic)\n",
      raw);
    return 0;
  }

  if (boxoban_difficulty_name_from_id(parsed) == NULL) {
    fprintf(stderr,
      "Boxoban 'difficulty' int must be in [0, 4], got %d. Falling back to 0 (basic)\n",
      parsed);
    return 0;
  }

  return parsed;
}

void my_init(Env* env, Dict* kwargs) {
  env->difficulty_id = parse_difficulty_id(kwargs);
  env->size = (int)dict_get(kwargs, "size")->value;
  env->max_steps = (int)dict_get(kwargs, "max_steps")->value;
  env->int_r_coeff = (float)dict_get(kwargs, "int_r_coeff")->value;
  env->target_loss_pen_coeff = (float)dict_get(kwargs, "target_loss_pen_coeff")->value;
  env->num_agents = 1;
  init(env);
}



void my_log(Log* log, Dict* out) {
  dict_set(out, "perf", log->perf);
  dict_set(out, "score", log->score);
  dict_set(out, "episode_return", log->episode_return);
  dict_set(out, "episode_length", log->episode_length);
  dict_set(out, "targets_hit", log->on_targets);
  dict_set(out, "difficulty", log->difficulty);
}
