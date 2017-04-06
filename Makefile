CC=mpicc
CLFAGS=
SOLVED_C=${wildcard mpi_solved*.c}
SOLVED=${SOLVED_C:.c=}
EXECS=${SOLVED}

all: ${EXECS}

${SOLVED} : % : %.c
	${CC} ${CFLAGS} $^ -o $@

clean:
	rm -f ${EXECS}
