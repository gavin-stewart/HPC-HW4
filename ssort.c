/* Parallel sample sort
 */
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <stdlib.h>

static int compare(const void *a, const void *b)
{
  int *da = (int *)a;
  int *db = (int *)b;

  if (*da > *db)
    return 1;
  else if (*da < *db)
    return -1;
  else
    return 0;
}

int main( int argc, char *argv[])
{
  int rank, num_cores, num_splitters;
  int i, j, N, num_samples, bucket, total_to_recv;
  int *vec, *samples, *recv_samples, *splitters, *num_to_send, *num_to_recv, 
        *send_offsets, *recv_offsets, *final_vec;
  double t_start, t_end;
  char filename[256];
  FILE *write_file;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_cores);

  if(argc < 2) {
    if(rank == 0) {
      printf("Usage: %s ints_per_thread\n", argv[0]);
    }
    MPI_Finalize();
    return EXIT_SUCCESS;
  }

  /* Number of random numbers per processor (this should be increased
   * for actual tests or could be passed in through the command line */
  N = atoi(argv[1]);

  sprintf(filename, "sorted_r%d_N%d.txt", rank, N);

  vec = calloc(N, sizeof(int));
  /* seed random number generator differently on every core */
  srand((unsigned int) (rank + 393919));

  /* fill vector with random integers */
  for (i = 0; i < N; ++i) {
    vec[i] = rand(); 
  }
  //printf("rank: %d, first entry: %d\n", rank, vec[0]);
  if(rank == 0) {
    t_start = MPI_Wtime();
  }

  /* sort locally */
  qsort(vec, N, sizeof(int), compare);

  /* randomly sample s entries from vector or select local splitters,
   * i.e., every N/P-th entry of the sorted vector */
  num_samples = N / num_cores;
  if(num_samples == 0) { //Ensure each core passes at least one sample.
    num_samples++;
  }
  samples = malloc(num_samples * sizeof(int));
  for(i = 0; i < num_samples; i++) {
    samples[i] = vec[i * num_cores];
  }

  

  /* every processor communicates the selected entries
   * to the root processor; use for instance an MPI_Gather */
  if(rank == 0) {
    recv_samples = calloc(num_samples * num_cores, sizeof(int));
  }
  MPI_Gather(samples, num_samples, MPI_INT, recv_samples, 
            num_samples, MPI_INT, 0, MPI_COMM_WORLD);

  /* root processor does a sort, determinates splitters that
   * split the data into P buckets of approximately the same size */
  num_splitters = num_cores - 1;
  splitters = calloc(num_splitters, sizeof(int));
  if(rank == 0) {
    qsort(recv_samples, num_samples * num_cores, sizeof(int), compare);
    for(i = 0; i < num_splitters; i++) {
      splitters[i] = recv_samples[(i+1) * num_samples];
    }
  }

  /* root process broadcasts splitters */
  MPI_Bcast(splitters, num_splitters, MPI_INT, 0, MPI_COMM_WORLD);

  /* every processor uses the obtained splitters to decide
   * which integers need to be sent to which other processor (local bins) */
  num_to_send = calloc(num_cores, sizeof(int));
  i = 0;
  for(bucket = 0; bucket < num_cores; bucket++) {
    for(; (i < N); i++) {
      if((bucket < num_cores - 1) && vec[i] > splitters[bucket]) {
        break;
      }
      num_to_send[bucket]++;
    }
  }

  /* send and receive: either you use MPI_AlltoallV, or
   * (and that might be easier), use an MPI_Alltoall to share
   * with every processor how many integers it should expect,
   * and then use MPI_Send and MPI_Recv to exchange the data */
  num_to_recv = malloc(num_cores * sizeof(int));
  MPI_Alltoall(num_to_send, 1, MPI_INT, num_to_recv, 1, 
               MPI_INT, MPI_COMM_WORLD);
  total_to_recv = 0;
  for(i = 0; i < num_cores; i++) {
    total_to_recv += num_to_recv[i];
  }

  send_offsets = calloc(num_cores, sizeof(int));
  recv_offsets = calloc(num_cores, sizeof(int));

  total_to_recv = 0;

  for(i = 0; i < num_cores; i++) {
    total_to_recv += num_to_recv[i];
    for(j = 0; j < i; j++) {
      send_offsets[i] += num_to_send[j];
      recv_offsets[i] += num_to_recv[j];
    }
  }

  final_vec = malloc(total_to_recv * sizeof(int));

  MPI_Alltoallv(vec, num_to_send, send_offsets, MPI_INT, 
                final_vec, num_to_recv, recv_offsets, MPI_INT, MPI_COMM_WORLD);


  /* do a local sort */
  qsort(final_vec, total_to_recv, sizeof(int), compare);

  /* every processor writes its result to a file */
  write_file = fopen(filename, "w+");
  for(i = 0; i < total_to_recv; i++) {
    fprintf(write_file, "%d,", final_vec[i]);
  }
  fclose(write_file);

  if(rank == 0) {
    t_end = MPI_Wtime();
    printf("%d ints sorted in %f seconds.\n", N * num_cores, t_end - t_start);
  }

  free(vec);
  free(samples);
  if(rank == 0) {
    free(recv_samples);
  }
  free(splitters);
  free(num_to_send);
  free(num_to_recv);
  free(send_offsets);
  free(recv_offsets);
  free(final_vec);
  MPI_Finalize();
  return 0;
}
