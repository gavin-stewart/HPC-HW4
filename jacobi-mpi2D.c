#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"

#define TOP 0
#define RIGHT 1
#define BOTTOM 2
#define LEFT 3

#define TOL (1e-4)
#define TAG 999

//#define PRINT_GRID

/*
 * A data structure which holds the MPI_Request object as well as the buffer
 * used to send or receive the data.
 */
typedef struct Ghost_layer_request {
    double *data;
    MPI_Request request;
} Ghost_layer_request;

/*
 * Data about the location and problem for a specific MPI core.
 */
typedef struct Core_data {
    int Nl;
    int row;
    int column;
    int cores_per_edge;
    double h_sqr;
} Core_data;

/*
 * A structure containing information about a node and the communication 
 * requests (either send or receive) it makes.
 */
typedef struct Ghost_data {
    Core_data *cd;
    Ghost_layer_request *top;
    Ghost_layer_request *right;
    Ghost_layer_request *bottom;
    Ghost_layer_request *left;
} Ghost_data;

int is_top(Core_data *);
int is_bottom(Core_data *);
int is_left(Core_data *);
int is_right(Core_data *);

void send_layer_data(double *, Core_data*, Ghost_data*);
void request_layer_data(Ghost_data *, Core_data*);
void update_ghost_layers(double*, Ghost_data*);
void wait_for_request(Ghost_data *);
void free_request_data(Ghost_data *);
void jacobi_update(double *, double *, Core_data*);
void setup_ghost_data(Ghost_data*, Core_data*);
void request_layer(int, int, Ghost_layer_request*);
void print_grid(double*, Core_data*); 

double compute_residual(double*, Core_data *); 

int main(int argc, char **argv) {
    int N, Nl, p, max_iter, iter;
    int rank, num_cores, row_ind, col_ind, cores_per_edge;
    double *u, *unew;
    double resid, start_resid, t_start, t_end;
    Core_data *core_data;
    
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_cores);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if(argc < 3) {
        if(rank == 0) {
            printf("Usage: %s #gridpoints #iterations\n", argv[0]);
        }
        MPI_Finalize();
        return EXIT_SUCCESS;
    }

    //Get number of gridpoints, number of iterations
    p = (int)floor(log(num_cores) / log(4));
    Nl = atoi(argv[1]);
    max_iter = atoi(argv[2]);
    cores_per_edge = 1 << p; //2^p
    N = Nl * cores_per_edge;
    if (cores_per_edge * cores_per_edge != num_cores) {
        if(rank == 0) {
            printf("Expected the number of cores to be a power of 4. Exiting. . .\n");
        }
        MPI_Finalize();
        return EXIT_SUCCESS;
    }

    // Compute the row and column of the core, as well as what edge(s) it 
    row_ind = rank % cores_per_edge;
    col_ind = rank / cores_per_edge;

    //Initializations
    core_data = malloc(sizeof(Core_data));
    u = calloc((Nl + 2) * (Nl + 2), sizeof(double));
    unew = calloc((Nl + 2) * (Nl + 2), sizeof(double));
    core_data->cores_per_edge = cores_per_edge;
    core_data->Nl = Nl; 
    core_data->row = row_ind;
    core_data->column = col_ind;
    core_data->h_sqr = 1.0 / (N + 1);
    core_data->h_sqr *= core_data->h_sqr;

    if(rank == 0) {
        t_start = MPI_Wtime();
    }

    start_resid = compute_residual(u, core_data); 
    resid = start_resid;

    #ifdef PRINT_RESID
    if(rank == 0) {
        printf("Starting residual: %f\n", start_resid);
    }
    #endif

    for(iter = 0; iter < max_iter && resid >= TOL * start_resid; iter++) {
        jacobi_update(u, unew, core_data);
        
        double *tmp = u;
        u = unew;
        unew = tmp;
        
        #ifdef PRINT_GRID
        int print_rank;
        for(print_rank = 0; print_rank < num_cores; print_rank++) {
            if(rank == print_rank) {
                print_grid(u, core_data);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
        #endif

        resid = compute_residual(u, core_data);
    
        #ifdef PRINT_GRID
        for(print_rank = 0; print_rank < num_cores; print_rank++) {
            if(rank == print_rank) {
                print_grid(u, core_data);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
        #endif

        #ifdef PRINT_RESID
        if(rank == 0) {
            printf("Iter %d: Residual = %f\n", iter, resid);
        }
        #endif
    }
    
    if(rank == 0) {
        t_end = MPI_Wtime();
        printf("Elapsed time for %d processors: %f seconds\n\
\t(Nl = %d, N=%d, iter=%d)\n",num_cores, t_end - t_start, Nl, N, iter); 
    }

    free(u);
    free(unew);
    free(core_data);
    MPI_Finalize();
    return EXIT_SUCCESS;
}

/*
 * Using the data in u, all cores perform a Jacobi update and put the new 
 * iterate in unew.
 */
void jacobi_update(double *u, double *unew, Core_data *cd) {
    int i,j,index,Nl;
    double h_sqr;
    Ghost_data *send_layers, *recv_layers;
    send_layers = malloc(sizeof(Ghost_data));
    recv_layers = malloc(sizeof(Ghost_data));
    h_sqr = cd->h_sqr;
    Nl = cd->Nl;
    
    send_layer_data(u, cd, send_layers);
    request_layer_data(recv_layers, cd);

    //Perform the inner updates
    for(i = 1; i < Nl - 1; i++) {
        for(j = 1; j < Nl - 1; j++) {
            index = (Nl + 2) * (i+1) + j + 1;
            unew[index] = (h_sqr + u[index - (Nl+2)] + u[index - 1] \
                        + u[index + 1] + u[index + (Nl+2)]) / 4;
        }
    }

    //Wait for the receive request to clear
    wait_for_request(recv_layers);
    update_ghost_layers(u, recv_layers);
    
    //Perform the calculations using the ghost nodes
    //Top
    for(i = 0; i < Nl; i++) {
        index = (Nl + 2) + 1 + i; 
        unew[index] = (h_sqr + u[index - (Nl+2)] + u[index - 1] \
                        + u[index + 1] + u[index + (Nl+2)]) / 4;
    }
    //Bottom
    for(i = 0; i < Nl; i++) {
        index = (Nl + 2) * Nl + 1 + i;
        unew[index] = (h_sqr + u[index - (Nl+2)] + u[index - 1] \
                        + u[index + 1] + u[index + (Nl+2)]) / 4;
    }
    //Left
    for(i = 1; i < Nl-1; i++) {
        index = (Nl + 2) * (i+1) + 1;
        unew[index] = (h_sqr + u[index - (Nl+2)] + u[index - 1] \
                        + u[index + 1] + u[index + (Nl+2)]) / 4;
    }
    //Right
    for(i = 1; i < Nl-1; i++) {
        index = (Nl + 2) * (i+1) + Nl;
        unew[index] = (h_sqr + u[index - (Nl+2)] + u[index - 1] \
                        + u[index + 1] + u[index + (Nl+2)]) / 4;
    }

    free_request_data(recv_layers);
    free(recv_layers);
    //Make sure other tasks receive their data before cleaning up.
    wait_for_request(send_layers); 
    free_request_data(send_layers);
    free(send_layers);
}

/*
 * All cores compute the residual for the current iterate u.
 */
double compute_residual(double* u, Core_data *cd) { 
    double resid, tmp, h_sqr;
    int i, j, index, Nl;

    Nl = cd->Nl;
    h_sqr = cd->h_sqr;
    Ghost_data *send_layers, *recv_layers;
    send_layers = malloc(sizeof(Ghost_data));
    recv_layers = malloc(sizeof(Ghost_data));
    send_layer_data(u, cd, send_layers);
    request_layer_data(recv_layers, cd);
    
    resid = 0;

    //Perform the calculations on the inner gridpoints
    for(i = 1; i < Nl - 1; i++) {
        for(j = 1; j < Nl - 1; j++) {
            index = (Nl + 2) * (i+1) + j + 1;
            tmp = 1 + (u[index - (Nl+2)] + u[index - 1] - 4 * u[index] \
                        + u[index + 1] + u[index + (Nl+2)]) / h_sqr;
            resid += tmp * tmp;
        } 
    }

    //Wait for the receive requests to clear
    wait_for_request(recv_layers);

    update_ghost_layers(u, recv_layers);

    //Perform the calculations using the ghost nodes
    //Top
    for(i = 0; i < Nl; i++) {
        index = (Nl + 2) + 1 + i; 
        tmp = 1 + (u[index - (Nl+2)]  + u[index - 1] - 4 * u[index] \
                    + u[index + 1] + u[index + Nl + 2]) / h_sqr;
        resid += tmp * tmp;
    }
    //Bottom
    for(i = 0; i < Nl; i++) {
                index = (Nl + 2) * Nl + 1 + i;
        tmp = 1 + (u[index - (Nl+2)]  + u[index - 1] - 4 * u[index] \
                    + u[index + 1] + u[index + Nl + 2]) / h_sqr;
        resid += tmp * tmp;
    }
    //Left
    for(i = 1; i < Nl-1; i++) {
        index = (Nl + 2) * (i+1) + 1;
        tmp = 1 + (u[index - (Nl+2)]  + u[index - 1] - 4 * u[index] \
                    + u[index + 1] + u[index + Nl + 2]) / h_sqr;
        resid += tmp * tmp;
    }
    //Right
    for(i = 1; i < Nl-1; i++) {
        index = (Nl + 2) * (i+2) - 2;
        tmp = 1 + (u[index - (Nl+2)]  + u[index - 1] - 4 * u[index] \
                    + u[index + 1] + u[index + Nl + 2]) / h_sqr;
        resid += tmp * tmp;
    }


    // Perform a reduction to get the total residual
    MPI_Allreduce(&resid, &tmp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    resid = sqrt(tmp);



    free_request_data(send_layers);
    free_request_data(recv_layers);
    free(send_layers);
    free(recv_layers);
    return resid;
}

/*
 * Send a layer (packaged as a Ghost_layer_request) to the core with rank dest.
 * This function uses nonblocking sends.
 */
void send_layer(int dest, int Nl, Ghost_layer_request *glr) {
    MPI_Isend(glr->data, Nl, MPI_DOUBLE, dest, TAG, MPI_COMM_WORLD, 
                &(glr->request));
}

/*
 * Determine what data the core needs to send and make nonblocking calls to \
 * send that data.
 */
void send_layer_data(double *u, Core_data *cd, Ghost_data *gd){
    setup_ghost_data(gd, cd);
    int dest, row, column, cpe, Nl, i;
    row = cd->row;
    column = cd->column;
    cpe = cd->cores_per_edge;
    Nl = cd->Nl;
    if(!is_top(cd)) {
        for(i = 0; i < Nl; i++) {
            int u_index = Nl + 2 + 1 + i; 
            gd->top->data[i] = u[u_index];
        }
        dest = (row - 1) * cpe + column;
        send_layer(dest, Nl, gd->top);
    }
    if(!is_bottom(cd)) {
        for(i = 0; i < Nl; i++) {
            int u_index = (Nl + 2) * Nl + 1 + i; 
            gd->bottom->data[i] = u[u_index];
        }
        dest = (row + 1) * cpe + column;
        send_layer(dest, Nl, gd->bottom);
    }
    if(!is_left(cd)) {
        for(i = 0; i < Nl; i++) {
            int u_index = (Nl + 2) * (i+1) + 1; 
            gd->left->data[i] = u[u_index];
        }
        dest = row * cpe + column - 1;
        send_layer(dest, Nl, gd->left);
    }
    if(!is_right(cd)) {
        for(i = 0; i < Nl; i++) {
            int u_index = (Nl + 2)*(i+1) + Nl; 
            gd->right->data[i] = u[u_index];
        }
        dest = row * cpe + column + 1;
        send_layer(dest, Nl, gd->right);
    }
}

/*
 * Based on the core's position and the grid size, initialize the needed 
 * Ghost_layer_requests and their data buffers.
 */
void setup_ghost_data(Ghost_data *gd, Core_data *cd){
    int Nl;
    gd->cd = cd;
    Nl = cd->Nl;
    if(!is_top(cd)) {
        gd->top = malloc(sizeof(Ghost_layer_request));
        gd->top->data = malloc(Nl * sizeof(double));
    }
    if(!is_bottom(cd)) {
        gd->bottom = malloc(sizeof(Ghost_layer_request));
        gd->bottom->data = malloc(Nl * sizeof(double));
    }

    if(!is_left(cd)) {
        gd->left = malloc(sizeof(Ghost_layer_request));
        gd->left->data = malloc(Nl * sizeof(double));
    }

    if(!is_right(cd)) {
        gd->right = malloc(sizeof(Ghost_layer_request));
        gd->right->data = malloc(Nl * sizeof(double));
    }
}

/*
 * Request all of the data the needed to determine the ghost layers for the
 * Jacobi method.  The requests are placed in the Ghost_data object gd.
 */
void request_layer_data(Ghost_data *gd, Core_data *cd) {
    setup_ghost_data(gd, cd);
    int src, row, column, cpe, Nl;
    row = cd->row;
    column = cd->column;
    cpe = cd->cores_per_edge;
    Nl = cd->Nl;
    if(!is_top(cd)) {
        src = (row - 1) * cpe + column;
        request_layer(src, Nl, gd->top);
    }
    if(!is_bottom(cd)) {
        src = (row + 1) * cpe + column;
        request_layer(src, Nl, gd->bottom);
    }
    if(!is_left(cd)) {
        src = row * cpe + column - 1;
        request_layer(src, Nl, gd->left);
    }
    if(!is_right(cd)) {
        src = row * cpe + column + 1;
        request_layer(src, Nl, gd->right);
    }
}

/*
 * Request a ghost layer from src, and store the request data in glr.
 */
void request_layer(int src, int Nl, Ghost_layer_request *glr) {
    MPI_Irecv(glr->data, Nl, MPI_DOUBLE, src, TAG, MPI_COMM_WORLD, 
                &(glr->request));
}

/*
 * Write the data in the relevant buffers of gd to the edges of u.
 */
void update_ghost_layers(double *u, Ghost_data *gd) {
    Core_data *cd = gd->cd;
    int i, Nl;
    Nl = cd->Nl;
    if(!is_top(cd)) {
        for(i = 0; i < Nl; i++) {
            int u_index = 1 + i; 
            u[u_index] = gd->top->data[i];
        }
    }
    if(!is_bottom(cd)) {
        for(i = 0; i < Nl; i++) {
            int u_index = (Nl + 2) * (Nl+1) + 1 + i; 
            u[u_index] = gd->bottom->data[i];
        }
    }
    if(!is_left(cd)) {
        for(i = 0; i < Nl; i++) {
            int u_index = (Nl + 2) * (i+1); 
            u[u_index] = gd->left->data[i];
        }
    }
    if(!is_right(cd)) {
        for(i = 0; i < Nl; i++) {
            int u_index = (Nl + 2)*(i+1) + Nl + 1; 
            u[u_index] = gd->right->data[i];
        }
    }
}

/*
 * Wait for the requests in gd to finish sending/receiving.  Note that only the 
 * requests the core needs to make based on its position will be considered (eg
 * a core on the top edge will not wait for a nonexistant top request to 
 * finish).
 */
void wait_for_request(Ghost_data *gd) {
    Core_data *cd = gd->cd;
    if(!is_top(cd)) {
        MPI_Wait(&(gd->top->request), MPI_STATUS_IGNORE);
    }
    if(!is_left(cd)) {
        MPI_Wait(&(gd->left->request), MPI_STATUS_IGNORE);
    }
    if(!is_right(cd)) {
        MPI_Wait(&(gd->right->request), MPI_STATUS_IGNORE);
    }
    if(!is_bottom(cd)) {
        MPI_Wait(&(gd->bottom->request), MPI_STATUS_IGNORE);
    }
}

/*
 * Free the MPI_request and data buffer associated with glr.
 */
void free_layer_request(Ghost_layer_request * glr) {
    if(glr->request != MPI_REQUEST_NULL) {
        MPI_Request_free(&(glr->request));
    }
    free(glr->data);
}

/*
 * Print the grid (physical and ghost layers) for a given core.  This method is
 * used for debugging only.
 */
void print_grid(double *u, Core_data *cd) {
    int Nl = cd->Nl;
    int i,j;
    printf("\n(%d, %d):\n", cd->row, cd->column);
    for(i = 0; i < Nl+2; i++) {
        for(j = 0; j < Nl+2; j++) {
            printf("%f ", u[(Nl + 2) * i + j]);
        }
        printf("\n");
    }
}

/*
 * Free all Ghost_layer_requests in gd.
 */
void free_request_data(Ghost_data *gd) {
    Core_data *cd = gd->cd;
    if(!is_top(cd)) {
        free_layer_request(gd->top);
    }
    if(!is_bottom(cd)) {
        free_layer_request(gd->bottom);
    }
    if(!is_left(cd)) {
        free_layer_request(gd->left);
    }
    if(!is_right(cd)) {
        free_layer_request(gd->right);
    }
}

int is_top(Core_data *cd) {
   return cd->row == 0; 
}

int is_bottom(Core_data *cd) {
    return cd->row == cd->cores_per_edge - 1;
}

int is_left(Core_data *cd) {
    return cd->column == 0;
}

int is_right(Core_data *cd) {
    return cd->column == cd->cores_per_edge - 1;
}
