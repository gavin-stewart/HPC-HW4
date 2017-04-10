#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"

#define TOP_MASK (0x1)
#define RIGHT_MASK (0x2)
#define BOTTOM_MASK (0x4)
#define LEFT_MASK (0x8)

#define TOL (1e-4)
#define TAG 999

typedef struct Ghost_layer_request {
    double *data;
    MPI_Request request;
} Ghost_layer_request;

typedef struct Ghost_data{
    int Nl;
    int edge_code;
    Ghost_layer_request *top;
    Ghost_layer_request *right;
    Ghost_layer_request *bottom;
    Ghost_layer_request *left;
} Ghost_data;

int is_top(Ghost_data *g);
int is_right(Ghost_data *g);
int is_left(Ghost_data *g);
int is_bottom(Ghost_data *g);

void sent_receive_edge_data(char, double, double);
void jacobi_update(double *, double *, double, Ghost_data *); 
void inner_jacobi_update(double *, double *, double, int);
void outer_jacobi_update(double *, double *, double, Ghost_data *);
void communicate_ghost_data(int, int, double*, Ghost_data*, Ghost_data *);
void initialize_ghost_layer_data(Ghost_data *, char, int);
void finalize_ghost_layer_data(Ghost_data*);
void update_ghost_layer_data(Ghost_data*, double*);
void wait_for_requests(Ghost_data*);
void initialize_ghost_layer_request(Ghost_layer_request*, int);
void finalize_ghost_layer_request(Ghost_layer_request *);
void communicate_ghost_data_partner(int, Ghost_layer_request*, 
                                    Ghost_layer_request*, int);

int main(int argc, char **argv) {
    int N, Nl, p, max_iter, iter;
    int rank, num_cores, row_ind, col_ind;
    char edge_code = 0;
    double *u, *unew;
    double resid, start_resid;
    Ghost_data send_ghosts;
    Ghost_data recv_ghosts;
    if(argc < 3) {
        printf("Usage: %s #gridpoints #iterations\n", argv[0]);
        return EXIT_SUCCESS;
    }
    //Get number of gridpoints, number of iterations

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_cores);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    p = (int)floor(log(num_cores) / log(4));
    N = atoi(argv[1]);
    max_iter = atoi(argv[2]);

    if (1 << (2 * p) != num_cores) {
        if(rank == 0) {
            printf("Expected the number of cores to be a power of 4. Exiting. . .\n");
        }
        MPI_Finalize();
        return EXIT_SUCCESS;
    }

    Nl = N / (1 << p); //Nl = N / (2^p)


    /* 
     * Compute the row and column of the core, as well as what edge(s) it 
     * lies on, if any.
     */

    edge_code = 0;
    int edge_size = 1 << p; //2^p
    row_ind = rank % edge_size;
    col_ind = rank / edge_size;
    if(col_ind == 0) {
        edge_code |= TOP_MASK;
    }
    if(col_ind == edge_size - 1) {
        edge_code |= BOTTOM_MASK;
    }
    if(row_ind == 0) {
        edge_code |= LEFT_MASK;
    } 
    if(row_ind == edge_size - 1) {
        edge_code |= RIGHT_MASK;
    }

    //Initializations
    initialize_ghost_layer_data(&send_ghosts, edge_code, Nl);
    initialize_ghost_layer_data(&recv_ghosts, edge_code, Nl);
    u = calloc(Nl * Nl, sizeof(double));
    unew = calloc(Nl * Nl, sizeof(double));

    start_resid = N;
    resid = start_resid;

    for(iter = 0; iter < max_iter && resid >= TOL * start_resid; iter++) {
        //Communicate
        communicate_ghost_data(rank, edge_size, u, &send_ghosts, &recv_ghosts);

        //Perform the Jacobi update
        
        //Wait for all Isends to complete
        wait_for_requests(&send_ghosts);
        printf("Comm round %d done.\n", iter);    
    
        //Compute the residual

    }

    
    MPI_Finalize();
    return EXIT_SUCCESS;
}

void jacobi_update(double *u, double *unew, double h_sqr, Ghost_data *recv_ghosts) {
    inner_jacobi_update(u, unew, h_sqr, recv_ghosts->Nl);
    wait_for_requests(recv_ghosts);
    outer_jacobi_update(u, unew, h_sqr, recv_ghosts);
}

void inner_jacobi_update(double *u, double *unew, double h_sqr, int Nl) {
    int i,j, index;
    for(i = 1; i < Nl - 1; i++) {
        for(j = 1; j < Nl - 1; j++) {
            index = Nl * i + j;
            unew[index] = (1 / h_sqr + u[index - 1] + u[index + 1] \
                            + u[index - Nl] + u[index + Nl]) / 4;
        }
    }
}

void outer_jacobi_update(double *u, double *unew, double h_sqr, 
                        Ghost_data * g) {
    int i, index, Nl;
    Nl = g->Nl;
    
    //Top updates.
    index = 0;
    unew[index] = (1/h_sqr + u[index + 1] + u[index + Nl]) / 4;
    for(i = 1; i < Nl - 1; i++) {
        index = i;
        unew[index] = (1 / h_sqr + u[index + 1] + u[index - 1]\
                         + u[index + Nl]) / 4;
    }
    index = Nl - 1;
    unew[index] = (1/h_sqr + u[index - 1] + u[index + Nl]) / 4;
    //Factor in ghost layers.
    if(!is_top(g)) {
        for(i = 0; i < Nl; i++) {
            unew[i] += g->top->data[i] / 4;
        }
    }

    //Bottom updates
    index = Nl * (Nl - 1);
    unew[index] = (1/h_sqr + u[index + 1] + u[index - Nl]) / 4;
    for(i = 1; i < Nl - 1; i++) {
        index = i;
        unew[index] = (1 / h_sqr + u[index + 1] + u[index - 1]\
                         + u[index - Nl]) / 4;
    }
    index = Nl * Nl - 1;
    unew[index] = (1/h_sqr + u[index - 1] + u[index + Nl]) / 4;
    // Factor in ghost layers
    if(!is_bottom(g)) {
        for(i = 0; i < Nl; i++) {
            index = Nl * (Nl - 1) + 1;
            unew[index] += g->bottom->data[i];
        }
    }

    //Right updates
    for(i = 1; i < Nl - 1; i++) {
        
    }
    
    
    
    //Left updates
    
}

void communicate_ghost_data(int rank, int core_per_row, double *u, 
                            Ghost_data *send_ghosts, Ghost_data *recv_ghosts) {
    int partner;
    update_ghost_layer_data(send_ghosts, u);
    if(!is_top(send_ghosts)) {
        partner = rank - core_per_row;
        communicate_ghost_data_partner(partner, send_ghosts->top, 
                                        recv_ghosts->top, send_ghosts->Nl); 
    }
    if(!is_bottom(send_ghosts)) {
        partner = rank + core_per_row;
        communicate_ghost_data_partner(partner, send_ghosts->bottom, 
                                        recv_ghosts->bottom, send_ghosts->Nl); 
    }
    if(!is_left(send_ghosts)) {
        partner = rank - 1;
        communicate_ghost_data_partner(partner, send_ghosts->left, 
                                        recv_ghosts->left, send_ghosts->Nl); 
    }
    if(!is_right(send_ghosts)) {
        partner = rank + 1;
        communicate_ghost_data_partner(partner, send_ghosts->right, 
                                        recv_ghosts->right, send_ghosts->Nl); 
    }
}

void communicate_ghost_data_partner(int partner, Ghost_layer_request *send_glr,
                                    Ghost_layer_request *recv_glr, int Nl) {
        MPI_Isend(send_glr->data, Nl, MPI_DOUBLE, partner, TAG, MPI_COMM_WORLD,
                    &(send_glr->request));
        MPI_Irecv(recv_glr->data, Nl, MPI_DOUBLE, partner, TAG, MPI_COMM_WORLD,
                    &(recv_glr->request));
}

/*
 * Initialize the needed fields in the Ghost_data struct
 */
void initialize_ghost_layer_data(Ghost_data *sg, char edge_code, int Nl) {
    sg->edge_code = edge_code;
    sg->Nl = Nl;

    if(!is_top(sg)) {
        sg->top = malloc(sizeof(Ghost_layer_request));
        initialize_ghost_layer_request(sg->top, sg->Nl);
    }
    if(!is_bottom(sg)) {
        sg->bottom = malloc(sizeof(Ghost_layer_request));
        initialize_ghost_layer_request(sg->bottom, sg->Nl);
    }
    if(!is_left(sg)) {
        sg->left = malloc(sizeof(Ghost_layer_request));
        initialize_ghost_layer_request(sg->left, sg->Nl);
    }
    if(!is_right(sg)) {
        sg->right = malloc(sizeof(Ghost_layer_request));
        initialize_ghost_layer_request(sg->right, sg->Nl);
    }
}

/*
 * Finalize the fields in the Ghost_data struct
 */
void finalize_ghost_layer_data(Ghost_data *sg) {

}

/*
 * Update the ghost layer data to be sent from this node based on its vector u.
 */
void update_ghost_layer_data(Ghost_data *sg, double *u) {
    int j;
    if(!is_top(sg)) {
        memcpy(sg->top->data, u, sg->Nl * sizeof(double));
    }
    if(!is_bottom(sg)) {
        memcpy(sg->bottom->data, u + sg->Nl * (sg->Nl - 1), 
                sg->Nl * sizeof(double));
    }
    if(!is_left(sg)) {
        for(j = 0; j < sg->Nl; j++) {
            sg->left->data[j] = u[sg->Nl * j];
        }
    }
    if(!is_right(sg)) {
        for(j = 1; j <= sg->Nl; j++) {
            sg->right->data[j] = u[sg->Nl * j - 1];
        }
    }
}

void wait_for_requests(Ghost_data *g) {
    MPI_Status s;
    if(!is_top(g)) {
        MPI_Wait(&(g->top->request), &s);
    }
    if(!is_bottom(g)) {
        MPI_Wait(&(g->bottom->request), &s);
    }
    if(!is_left(g)) {
        MPI_Wait(&(g->left->request), &s);
    }
    if(!is_right(g)) {
        MPI_Wait(&(g->right->request), &s);
    }
}

/*
 * Initialize the needed fields in the Ghost_layer_request struct
 */
void initialize_ghost_layer_request(Ghost_layer_request *glr, int Nl) {
    glr->data = malloc(Nl * sizeof(double));
}

void finalize_ghost_layer_request(Ghost_layer_request *glr) {
    free(glr->data);
}

int is_top(Ghost_data *g) {
    return g->edge_code & TOP_MASK;
}

int is_bottom(Ghost_data *g) {
    return g->edge_code & BOTTOM_MASK;
}

int is_left(Ghost_data *g) {
    return g->edge_code & LEFT_MASK;
}

int is_right(Ghost_data *g) {
    return g->edge_code & RIGHT_MASK;
}
