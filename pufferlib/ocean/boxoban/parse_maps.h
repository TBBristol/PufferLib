#pragma once

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#define BOXOBAN_EXPECTED_ROWS 10
#define BOXOBAN_EXPECTED_COLS 10
#define BOXOBAN_PUZZLE_OBS_BYTES (4 * 10 * 10)
#define BOXOBAN_PUZZLE_META_BYTES 5
#define BOXOBAN_PUZZLE_BYTES (BOXOBAN_PUZZLE_OBS_BYTES + BOXOBAN_PUZZLE_META_BYTES)

static bool boxoban_file_exists(const char* path) {
    struct stat st;
    return path != NULL && stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

static bool boxoban_dir_exists(const char* path) {
    struct stat st;
    return path != NULL && stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

static int boxoban_ends_with_txt(const char* name) {
    size_t n = strlen(name);
    return n >= 4 && strcmp(name + n - 4, ".txt") == 0;
}

static int boxoban_mkdirs_for_file(const char* out_path) {
    char tmp[PATH_MAX];
    size_t len = strlen(out_path);
    if (len == 0 || len >= sizeof(tmp)) {
        return -1;
    }
    strcpy(tmp, out_path);
    for (size_t i = 1; i < len; i++) {
        if (tmp[i] == '/') {
            tmp[i] = '\0';
            if (mkdir(tmp, 0755) != 0 && errno != EEXIST) {
                return -1;
            }
            tmp[i] = '/';
        }
    }
    return 0;
}

typedef struct {
    char** items;
    int count;
    int cap;
} BoxobanPathList;

static int boxoban_run_python(const char* code) {
    char cmd[8192];
    int n = snprintf(cmd, sizeof(cmd), "python -c \"%s\"", code);
    if (n <= 0 || (size_t)n >= sizeof(cmd)) return -1;
    int rc = system(cmd);
    return rc == 0 ? 0 : -1;
}

static void boxoban_path_list_free(BoxobanPathList* list) {
    if (!list) return;
    for (int i = 0; i < list->count; i++) free(list->items[i]);
    free(list->items);
    list->items = NULL;
    list->count = 0;
    list->cap = 0;
}

static int boxoban_path_list_push(BoxobanPathList* list, const char* path) {
    if (list->count == list->cap) {
        int next_cap = list->cap == 0 ? 16 : list->cap * 2;
        char** next = (char**)realloc(list->items, (size_t)next_cap * sizeof(char*));
        if (!next) return -1;
        list->items = next;
        list->cap = next_cap;
    }
    char* copy = strdup(path);
    if (!copy) return -1;
    list->items[list->count++] = copy;
    return 0;
}

static int boxoban_cmp_path(const void* a, const void* b) {
    const char* const* pa = (const char* const*)a;
    const char* const* pb = (const char* const*)b;
    return strcmp(*pa, *pb);
}

static int boxoban_collect_txt_maps(const char* dir_path, BoxobanPathList* out) {
    DIR* dir = opendir(dir_path);
    if (!dir) return -1;

    struct dirent* ent;
    while ((ent = readdir(dir)) != NULL) {
        if (ent->d_name[0] == '.') continue;
        if (!boxoban_ends_with_txt(ent->d_name)) continue;

        char full[PATH_MAX];
        int n = snprintf(full, sizeof(full), "%s/%s", dir_path, ent->d_name);
        if (n <= 0 || (size_t)n >= sizeof(full)) {
            closedir(dir);
            return -1;
        }
        if (boxoban_path_list_push(out, full) != 0) {
            closedir(dir);
            return -1;
        }
    }
    closedir(dir);
    return 0;
}

static int boxoban_get_rel_for_difficulty(const char* difficulty, char rel[64]) {
    if (strcmp(difficulty, "basic") == 0) strcpy(rel, "basic/train");
    else if (strcmp(difficulty, "easy") == 0) strcpy(rel, "easy/train");
    else if (strcmp(difficulty, "medium") == 0) strcpy(rel, "medium/train");
    else if (strcmp(difficulty, "hard") == 0) strcpy(rel, "hard");
    else if (strcmp(difficulty, "unfiltered") == 0) strcpy(rel, "unfiltered/train");
    else return -1;
    return 0;
}

static int boxoban_ensure_text_maps(const char* difficulty) {
    char rel[64];
    if (boxoban_get_rel_for_difficulty(difficulty, rel) != 0) return -1;

    char level_dir[PATH_MAX];
    int dn = snprintf(level_dir, sizeof(level_dir), "pufferlib/ocean/boxoban/boxoban-levels/%s", rel);
    if (dn <= 0 || (size_t)dn >= sizeof(level_dir)) return -1;

    BoxobanPathList existing = {0};
    if (boxoban_dir_exists(level_dir) && boxoban_collect_txt_maps(level_dir, &existing) == 0 && existing.count > 0) {
        boxoban_path_list_free(&existing);
        return 0;
    }
    boxoban_path_list_free(&existing);

    if (strcmp(difficulty, "basic") == 0 || strcmp(difficulty, "easy") == 0) {
        char py[4096];
        int pn = snprintf(
            py, sizeof(py),
            "from pathlib import Path; "
            "from pufferlib.ocean.boxoban.generate_easy_maps import generate_basic_maps, generate_easy_maps; "
            "out=Path(r'%s'); "
            "out.mkdir(parents=True, exist_ok=True); "
            "%s(out)",
            level_dir,
            strcmp(difficulty, "basic") == 0 ? "generate_basic_maps" : "generate_easy_maps"
        );
        if (pn <= 0 || (size_t)pn >= sizeof(py)) return -1;
        return boxoban_run_python(py);
    }

    {
        char py[6144];
        int pn = snprintf(
            py, sizeof(py),
            "import shutil,tempfile,urllib.request,zipfile; "
            "from pathlib import Path; "
            "difficulty=r'%s'; "
            "base_url='https://raw.githubusercontent.com/TBBristol/pufferlib_boxoban_levels/main'; "
            "zip_url=f\"{base_url}/{difficulty}.zip\"; "
            "level_root=Path('pufferlib/ocean/boxoban/boxoban-levels'); "
            "level_root.mkdir(parents=True, exist_ok=True); "
            "tmp=Path(tempfile.mkdtemp()); "
            "zip_path=tmp/f\"{difficulty}.zip\"; "
            "urllib.request.urlretrieve(zip_url, zip_path); "
            "zf=zipfile.ZipFile(zip_path); zf.extractall(tmp); zf.close(); "
            "cand=None; "
            "paths=list(tmp.rglob(difficulty)); "
            "cand=next((p for p in paths if p.is_dir()), None); "
            "assert cand is not None, f\"Downloaded zip missing {difficulty} directory\"; "
            "dest=level_root/difficulty; "
            "shutil.copytree(cand, dest, dirs_exist_ok=True)",
            difficulty
        );
        if (pn <= 0 || (size_t)pn >= sizeof(py)) return -1;
        return boxoban_run_python(py);
    }
}

static int boxoban_encode_puzzle(char rows[BOXOBAN_EXPECTED_ROWS][BOXOBAN_EXPECTED_COLS + 1], uint8_t out[BOXOBAN_PUZZLE_BYTES]) {
    int agent_x = -1, agent_y = -1;
    int n_boxes = 0, n_targets = 0, on_target = 0;

    memset(out, 0, BOXOBAN_PUZZLE_BYTES);
    uint8_t* agent = out;
    uint8_t* walls = out + 100;
    uint8_t* boxes = out + 200;
    uint8_t* targ = out + 300;
    uint8_t* meta = out + 400;

    for (int r = 0; r < BOXOBAN_EXPECTED_ROWS; r++) {
        for (int c = 0; c < BOXOBAN_EXPECTED_COLS; c++) {
            char ch = rows[r][c];
            int idx = r * BOXOBAN_EXPECTED_COLS + c;
            bool is_agent = (ch == '@' || ch == '+');
            bool is_wall = (ch == '#');
            bool is_box = (ch == '$' || ch == '*');
            bool is_targ = (ch == '.' || ch == '*' || ch == '+');

            if (is_agent) {
                if (agent_x != -1) return -1;
                agent_x = c;
                agent_y = r;
            }

            n_boxes += (int)is_box;
            n_targets += (int)is_targ;
            on_target += (int)(is_box && is_targ);

            agent[idx] = (uint8_t)is_agent;
            walls[idx] = (uint8_t)is_wall;
            boxes[idx] = (uint8_t)is_box;
            targ[idx] = (uint8_t)is_targ;
        }
    }

    if (agent_x < 0) return -1;

    meta[0] = (uint8_t)agent_x;
    meta[1] = (uint8_t)agent_y;
    meta[2] = (uint8_t)n_boxes;
    meta[3] = (uint8_t)n_targets;
    meta[4] = (uint8_t)on_target;
    return 0;
}

static int boxoban_parse_file_into_bin(const char* txt_path, FILE* out, int* written_count) {
    FILE* f = fopen(txt_path, "r");
    if (!f) return -1;

    char line[512];
    char rows[BOXOBAN_EXPECTED_ROWS][BOXOBAN_EXPECTED_COLS + 1];
    int row_count = 0;
    int puzzle_idx = 0;

    while (fgets(line, sizeof(line), f) != NULL) {
        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
            line[--len] = '\0';
        }

        if (line[0] == ';') {
            row_count = 0;
            puzzle_idx++;
            continue;
        }

        bool all_ws = true;
        for (size_t i = 0; i < len; i++) {
            if (!isspace((unsigned char)line[i])) {
                all_ws = false;
                break;
            }
        }
        if (all_ws) continue;

        if (row_count >= BOXOBAN_EXPECTED_ROWS) {
            fprintf(stderr, "[Boxoban] Skipping malformed puzzle in %s puzzle#%d: too many rows\n", txt_path, puzzle_idx);
            row_count = 0;
            puzzle_idx++;
            continue;
        }

        if (len != BOXOBAN_EXPECTED_COLS) {
            fprintf(stderr, "[Boxoban] Skipping malformed puzzle in %s puzzle#%d: expected 10 cols, got %zu\n", txt_path, puzzle_idx, len);
            row_count = 0;
            puzzle_idx++;
            continue;
        }

        memcpy(rows[row_count], line, BOXOBAN_EXPECTED_COLS + 1);
        row_count++;
        if (row_count == BOXOBAN_EXPECTED_ROWS) {
            uint8_t buf[BOXOBAN_PUZZLE_BYTES];
            if (boxoban_encode_puzzle(rows, buf) != 0) {
                fprintf(stderr, "[Boxoban] Skipping malformed puzzle in %s puzzle#%d: invalid agent count\n", txt_path, puzzle_idx);
            } else if (fwrite(buf, 1, BOXOBAN_PUZZLE_BYTES, out) != BOXOBAN_PUZZLE_BYTES) {
                fclose(f);
                return -1;
            } else {
                (*written_count)++;
            }
            row_count = 0;
            puzzle_idx++;
        }
    }

    fclose(f);
    return 0;
}

static int boxoban_collect_maps_for_difficulty(const char* difficulty, BoxobanPathList* out) {
    char rel[64];
    if (boxoban_get_rel_for_difficulty(difficulty, rel) != 0) return -1;

    char dir_path[PATH_MAX];
    int n = snprintf(dir_path, sizeof(dir_path), "pufferlib/ocean/boxoban/boxoban-levels/%s", rel);
    if (n <= 0 || (size_t)n >= sizeof(dir_path)) return -1;

    if (!boxoban_dir_exists(dir_path) || boxoban_collect_txt_maps(dir_path, out) != 0 || out->count == 0) {
        boxoban_path_list_free(out);
        if (boxoban_ensure_text_maps(difficulty) != 0) return -1;
        if (!boxoban_dir_exists(dir_path)) return -1;
        if (boxoban_collect_txt_maps(dir_path, out) != 0) return -1;
    }

    if (out->count > 1) qsort(out->items, (size_t)out->count, sizeof(char*), boxoban_cmp_path);
    return out->count > 0 ? 0 : -1;
}

static int boxoban_generate_bin(const char* difficulty, const char* out_bin_path, int* out_count) {
    BoxobanPathList maps = {0};
    int written = 0;
    FILE* out = NULL;

    if (boxoban_collect_maps_for_difficulty(difficulty, &maps) != 0) {
        boxoban_path_list_free(&maps);
        return -1;
    }
    if (boxoban_mkdirs_for_file(out_bin_path) != 0) {
        boxoban_path_list_free(&maps);
        return -1;
    }

    out = fopen(out_bin_path, "wb");
    if (!out) {
        boxoban_path_list_free(&maps);
        return -1;
    }

    for (int i = 0; i < maps.count; i++) {
        if (boxoban_parse_file_into_bin(maps.items[i], out, &written) != 0) {
            fclose(out);
            boxoban_path_list_free(&maps);
            return -1;
        }
    }

    fclose(out);
    boxoban_path_list_free(&maps);

    if (written <= 0) return -1;
    if (out_count) *out_count = written;
    return 0;
}

static int boxoban_extract_difficulty_from_bin_path(const char* path, char out[32]) {
    const char* tag = "boxoban_maps_";
    const char* start = strstr(path, tag);
    if (!start) return -1;
    start += strlen(tag);
    const char* end = strstr(start, ".bin");
    if (!end || end <= start) return -1;
    size_t n = (size_t)(end - start);
    if (n >= 32) return -1;
    memcpy(out, start, n);
    out[n] = '\0';
    return 0;
}

static int boxoban_ensure_bin_for_path(const char* bin_path) {
    if (boxoban_file_exists(bin_path)) return 0;

    char difficulty[32];
    if (boxoban_extract_difficulty_from_bin_path(bin_path, difficulty) != 0) return -1;

    int count = 0;
    if (boxoban_generate_bin(difficulty, bin_path, &count) != 0) return -1;
    fprintf(stderr, "[Boxoban] Generated %d puzzles for '%s' at %s\n", count, difficulty, bin_path);
    return 0;
}
