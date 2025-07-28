/* Based on Jin, X., Duan, Z., Song, W., Li, Q., 2023. Container stacking optimization based on Deep Reinforcement Learning. 
Engineering Applications of Artificial Intelligence 123, 106508. https://doi.org/10.1016/j.engappai.2023.106508
*/


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "raylib.h"
#include <stdbool.h>
#include <time.h>
#include <math.h>

/* CONSTS */

#define EMPTY_SLOT -1
#define STACK(env, s, h) (env->stacks[(s) * (env->max_height) + (h)])
#define OBS(env, s, f) (env->observations[(s) * 6 + f])

// Required struct by pufferlib. Only use floats!
typedef struct {
    float perf; // Recommended 0-1 normalized single real number perf metric
    float score; // Recommended unnormalized single real number perf metric
    float episode_return; // Recommended metric: sum of agent rewards over episode
    float episode_length; // Recommended metric: number of steps of agent episode
    float num_invalids;
    float unsorteds;
    // Any extra fields you add here may be exported to Python in binding.c
    float n; // Required as the last field 
} Log;

typedef struct {
    Log log; // Required field. Env binding code uses this to aggregate logs
    float* observations; // Required. You can use any obs type, but make sure it matches in Python!
    int* actions; // Required. int* for discrete/multidiscrete, float* for box
    float* rewards; // Required
    unsigned char* terminals; // Required. We don't yet have truncations as standard yet
    int tick;
    // params
    int num_containers;
    int num_stacks;
    int max_height;
    int max_ep_steps;
    int seed;
    int unsorted;
    int next_container;
    int *container_leaving_priorities; // ptr to array of priorities for each container
    int *stacks; // ptr to array of stacks, each stack is an array of ints of size max_height
    float reward_us;
    float score;
    float num_invalids;
    float reward_max_breach;
    bool reset_max_breach;
}ContainerStacking;

void add_log(ContainerStacking* env) {
    env->log.perf += (env->rewards[0] > 0) ? 1 : 0;
    env->log.score += env->next_container - env->unsorted - env->num_invalids;;
    env->log.episode_length += env->tick;
    env->log.episode_return += env->next_container - env->unsorted - env->num_invalids;
    env->log.unsorteds += (float) env->unsorted;
    env->log.num_invalids += (float) env->num_invalids;
    env->log.n++;
}



/* -----------------------------------------------------------
 * Helpers
 * -----------------------------------------------------------*/

 void sample_unique_integers(int *out, int n) {
    // Fill with 0..n-1
    for (int i = 0; i < n; i++) {
        out[i] = i;
    }

    // Fisher-Yates shuffle
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = out[i];
        out[i] = out[j];
        out[j] = tmp;
    }
}


 void initialise_stacks(ContainerStacking *env) {
    if (env->stacks != NULL) {
        free(env->stacks);
    }
    env->stacks = malloc(sizeof(int) * env->num_stacks * env->max_height);

    for (int i = 0; i < env->num_stacks * env->max_height; i++) {
        env->stacks[i] = EMPTY_SLOT;
    }
 }

 void generate_container_priorities(ContainerStacking *env) {

    /*Generate container leaving priorities this is a list
    where element 0 is the priority of the first container
    to arrive. Lower values indicate higher priority.*/

    if (env->container_leaving_priorities != NULL) {
        free(env->container_leaving_priorities);
    }
    env->container_leaving_priorities = malloc(sizeof(int) * env->num_containers);

    sample_unique_integers(env->container_leaving_priorities, env->num_containers);
    }
 


bool free_space(ContainerStacking *env,int stack) {

  //assume any free space means top slot is EMPTY

  return STACK(env, stack, env->max_height -1) == EMPTY_SLOT;
}

int find_next_height(ContainerStacking *env,int stack) {
    for (int h=0; h < env->max_height; h++){
        if (STACK(env, stack, h) == EMPTY_SLOT) {
            return h;
        }
    }   
    return -1;
}

int find_height(ContainerStacking *env,int stack) {
    for (int h=0; h < env->max_height; h++){
        if (STACK(env, stack, h) == EMPTY_SLOT) {
            return h;
        }
    }
    return env->max_height;
    }

int count_unsorted_in_stack(ContainerStacking *env, int stack)
{
    int count = 0;
    for (int h = 1; h < env->max_height; h++) {
        if (STACK(env, stack, h) == EMPTY_SLOT)      // ⬅ break here
            break;
        if (STACK(env, stack, h) < STACK(env, stack, h - 1))
            count++;
    }
    return count;
}

int total_unsorted(ContainerStacking *env) {
    int count = 0;
    for (int s=0; s < env->num_stacks; s++){
        count += count_unsorted_in_stack(env, s);
    }
    return count;
}

int lowest_left(ContainerStacking *env) {
    int lowest = env->num_containers; // should be one higher than hiest possible generated

    //choice here we still need an obs if none left so if theres none left make it a really low priority
    if (env->next_container >= env-> num_containers){
        return 0;
    }
    for (int i = env->next_container; i < env->num_containers; i++) {
        if (env->container_leaving_priorities[i] < lowest) {
            lowest = env->container_leaving_priorities[i];
        }
    }
    return lowest;
}

int num_priority_less_than(ContainerStacking *env, int top){
    int count = 0;
    for (int i = env->next_container; i < env->num_containers; i++) {
        if (env->container_leaving_priorities[i] < top) {
            count +=1;
        }
    }
    return count;
}

int next_priority(ContainerStacking *env)
//choice here we still need an obs so if theres none left make it a really low priority
{
    if (env->next_container < env-> num_containers) {
        return env->container_leaving_priorities[env->next_container];
    }
    return env->num_containers;
}

bool stack_empty(ContainerStacking *env,int stack){
    if (STACK(env, stack, 0)== EMPTY_SLOT){
        return 1;
    }
    return 0;
}

int top_of_stack_priority(ContainerStacking *env,int stack){
    if (stack_empty(env,stack)){
        return 0; //effectively lowest priority so we want to stack here
    }
    return STACK(env, stack, find_next_height(env, stack) -1);
}

float linear_norm(int x, int xmax,int  xmin) {

    return (float) ((float) (x - xmin)/(float) (xmax-xmin));
}

float* generate_obs(ContainerStacking *env){
    
    int lowest_remaining = lowest_left(env);

    //only accurate after next container has been iterated
    int nxt_container_priority = next_priority(env);

    for (int s=0; s< env->num_stacks; s++) {

        float height_percent = (float) find_height(env,s)/ (float) env->max_height;

        int top_prior = top_of_stack_priority(env,s);

        int num_less_than_top = num_priority_less_than(env, top_of_stack_priority(env,s));

        int stack_unsorted = count_unsorted_in_stack(env, s);

        
        //printf("unsorted %d \n", stack_unsorted);
        //fflush(stdout);

        OBS(env, s, 0) = height_percent;
        OBS(env, s, 1) = linear_norm(nxt_container_priority, 0, env->num_containers-1);
        OBS(env, s, 2) = linear_norm(top_prior, 0, env->num_containers-1);
        OBS(env, s, 3) = linear_norm(lowest_remaining, 0, env->num_containers-1);
        OBS(env, s, 4) = linear_norm(num_less_than_top, 0, env->num_containers-1);
        OBS(env, s, 5) = linear_norm(stack_unsorted, env->max_height-1, 0);
    }
    return env->observations;
}


// Required function RESET

void c_reset(ContainerStacking *env) {

    //srand(((uintptr_t)env) ^ time(NULL)); //different seed per env and reset
  
    initialise_stacks(env);

    generate_container_priorities(env); //array each element int priority lowest num highest priority 

    //clear obs
    memset(env->observations, 0.0f, sizeof(float) * env->num_stacks * 6);

    env->tick = 0;
    env->unsorted = 0.0f;
    env-> next_container = 0;
    env->num_invalids = 0.0f;
    env->score = 0.0f;

}   
    

// Required Function Step

/* OBS shape = (n_stacks, 6).
          (1) height%
          (2) next container priority
          (3) top container priority
          (4) lowest remaining priority
          (5) # of remaining containers with priority < top
          (6) # of unsorted in this stack*/

void c_step(ContainerStacking *env) {

    env->terminals[0] = 0;
    env->rewards[0]   = 0.0f;
    env->tick += 1;

    //check max steps exceeded

    if (env->tick >= env->max_ep_steps) {
    env->terminals[0] = 1;
    env->rewards[0] = env->reward_max_breach; //penalises for total poss US - 1  to avoid it cheating
    add_log(env);
    c_reset(env);
    return;
}

    // Check if last container has been placed
    if (env->next_container >= env->num_containers) {
        env->terminals[0] = 1;
        env->rewards[0] = 10.0f; //penalise for unsorted every turn for agent learning but a bonus for finishing
        env->rewards[0] -= env->unsorted;
        add_log(env);
        c_reset(env);
        return;
    }

    int stack = env->actions[0]; // action is the stack to place the container in
    int old_unsorted = env-> unsorted;
    int container_priority = env->container_leaving_priorities[env->next_container];


    if (stack < 0 || stack >= env->num_stacks)  { //shouldn't ever hit this
        env->terminals[0] = 1;
        env-> rewards[0] = -100.0f;
        printf("ENV ACTION ERROR HIT");
        fflush(stdout);
        add_log(env);
        c_reset(env);
        return;
    }

    // Check if stack is valid
    if (!free_space(env, stack)){
        //env->terminals[0] = 1;
        env->rewards[0] = -1.0f; // neg reward for not placing a container
        env->log.num_invalids += 1.0f;
        //if (env->reset_max_breach) {
        //c_reset(env);
    //}
        return;
    }

    //Place container
    int h = find_next_height(env, stack);
    if (h == -1) {
        fprintf(stderr, "Error no free slot but stack checks as valid");
        exit(1);  
    }
    STACK(env, stack, h) = container_priority;
    env->next_container += 1;


    //new unsorted
    env->unsorted = total_unsorted(env);
    
    //set new OBS
    generate_obs(env);

    // Set reward 
    env-> rewards[0] = (float) (old_unsorted - env->unsorted); // effectively -1 for placing an US
    //printf("old unsorted %d, new unsorted %d\n", old_unsorted, env->unsorted);
    //fflush(stdout);

    env->rewards[0] += 1.0f; //one reward for placing a container

    /* no termination; step continues */
}


void c_close(ContainerStacking *env)
{
    (void)env;
    if (IsWindowReady()) {
        CloseWindow();
    }
}

void DrawDottedLineH(int x1, int x2, int y, int dotLength, int gapLength, Color color) {
    for (int x = x1; x < x2; x += dotLength + gapLength) {
        int end = x + dotLength;
        if (end > x2) end = x2;
        DrawLine(x, y, end, y, color);
    }
}

void DrawDottedLineV(int y1, int y2, int x, int dotLength, int gapLength, Color color) {
    for (int y = y1; y < y2; y += dotLength + gapLength) {
        int end = y + dotLength;
        if (end > y2) end = y2;
        DrawLine(x, y, x, end, color);
    }
}

#define MARGIN_X 20     // left margin
#define MARGIN_Y 20     // top margin
#define SKYBLUE    CLITERAL(Color){ 102, 191, 255, 255 }   // Sky Blue
#define WHITE      CLITERAL(Color){ 255, 255, 255, 255 }   // White


void c_render(ContainerStacking* env) {

    int gap;
    int cellw;
    int cellh;
    int font_size;
  
if (env->num_stacks > 50 || env->max_height > 40) {
    printf("WARNING: num_stacks or max_height too big for render must be env->num_stacks <= 49 and env->max_height <= 39\n");
    fflush(stdout);
    return;
}
    
else if (env->num_stacks > 24 || env->max_height > 20) {
    gap = 5;
    cellw = cellh = 10;
    font_size = 4;
}
else if (env->num_stacks > 12 || env->max_height > 9) {
    gap = 10;
    cellw = cellh = 20;
    font_size = 12;
}
else {
    gap = 20;
    cellw = cellh = 40;
    font_size = 20;
}
    
   
    const int screenW = 800, screenH = 600;

    int centre = screenW /2;
    


    if (!IsWindowReady()) {
        InitWindow(screenW, screenH, "PufferLib Stacking");
        SetTargetFPS(10);
    }

    // Standard across our envs so exiting is always the same
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    Font font = GetFontDefault();
    ClearBackground((Color){6, 24, 24, 255});


    int remaining_y = screenH - MARGIN_Y - cellh * 3;
    int remaining_x = MARGIN_X;

    int s_loc_x;
    bool odd = (env->num_stacks % 2) != 0;
    int half_floor = env->num_stacks / 2;           // e.g. 5 → 2

    if (odd) {
            s_loc_x = centre - cellw/2  - (cellw + gap) * half_floor;
        }
        else {
            s_loc_x = centre - (cellw + gap) * half_floor;
        } 

    DrawLine(MARGIN_X, remaining_y - gap, screenW - MARGIN_X, remaining_y - gap, WHITE);

    
    for (int i = env->next_container; i < env->num_containers; i++){

        DrawRectangle(remaining_x, remaining_y,
        cellw, cellh, SKYBLUE);

        char num[8];
        snprintf(num, sizeof num, "%d", env->container_leaving_priorities[i]);
        Vector2 size = MeasureTextEx(font, num, (float)font_size, 1.0f);
        int tx = remaining_x + (int)((cellw - size.x) / 2.0f);
        int ty = remaining_y + (int)((cellh - size.y) / 2.0f);
        DrawTextEx(font, num, (Vector2){ tx, ty }, (float)font_size, 1.0f, WHITE);
        
    
        remaining_x += cellw + gap;

        //fflush(stdout);

        if (remaining_x >= screenW - MARGIN_X - cellw) {
            remaining_y += gap + cellh;
            remaining_x = MARGIN_X;
        }
        if (remaining_y > screenH - MARGIN_Y - cellh) {
            DrawCircle(remaining_x + 3, remaining_y, 3, SKYBLUE);
            DrawCircle(remaining_x + 15, remaining_y, 3, SKYBLUE);
            DrawCircle(remaining_x + 27, remaining_y, 3, SKYBLUE);
            break;
        }

    }


    int s_loc_y    = screenH - MARGIN_Y - cellh * 4 - gap -2;

    
    int curr_p; //for checking unsorted
    

    int dot_y = screenH - MARGIN_Y - cellh * 3 - gap;
    int dot_y_top = dot_y - (2 + cellw) * env->max_height -1;


    for (int s = 0; s < env->num_stacks; ++s)             
    {   

        DrawDottedLineV(dot_y_top, dot_y, s_loc_x +1, 5,2, SKYBLUE);
        DrawDottedLineV(dot_y_top, dot_y, s_loc_x + cellw -1, 5,2, SKYBLUE);
        DrawDottedLineH(s_loc_x +1 , s_loc_x -1 + cellw, dot_y_top, 5, 2, SKYBLUE);



        if (!stack_empty(env,s)){
            curr_p = STACK(env,s,0);
        }


        int height = find_height(env, s);
        for (int h = 0; h < height; h++) {
            

            if (STACK(env,s,h) < curr_p) {
                DrawRectangle(s_loc_x, s_loc_y,
            cellw, cellh, RED);
            }
            else {
                DrawRectangle(s_loc_x, s_loc_y,
            cellw, cellh, SKYBLUE);
            }

            curr_p = STACK(env,s,h);

            char num[8];
            snprintf(num, sizeof num, "%d", STACK(env, s, h));
            Vector2 size = MeasureTextEx(font, num, (float)font_size, 1.0f);
            int tx = s_loc_x + (int)((cellw - size.x) / 2.0f);
            int ty = s_loc_y + (int)((cellh - size.y) / 2.0f);
            DrawTextEx(font, num, (Vector2){ tx, ty }, (float)font_size, 1.0f, WHITE);

            s_loc_y -= cellh + 2;

        }
        s_loc_y = screenH - MARGIN_Y - cellh * 4 - gap -2;
        s_loc_x += cellw + gap;
        
    }
    DrawText(TextFormat("Total Unsorted: %i", env->unsorted), 20, 20, font_size, WHITE);
    DrawText(TextFormat("Stack: %i", env->actions[0]), 20, 40, font_size, WHITE);


            
        EndDrawing();

}

