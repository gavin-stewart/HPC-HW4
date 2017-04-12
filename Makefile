CC=mpicc
CLFAGS=
SOLVED_C=${wildcard mpi_solved*.c}
SOLVED=${SOLVED_C:.c=}
EXECS=${SOLVED} jacobi-mpi2D ssort

all: ${EXECS}

${SOLVED} : % : %.c
	${CC} ${CFLAGS} $^ -o $@

jacobi-mpi2D : jacobi-mpi2D.c
	${CC} -lm ${CFLAGS} $^ -o $@

ssort : ssort.c
	${CC} ${CFLAGS} $^ -o $@

clean:
	rm -f ${EXECS}
