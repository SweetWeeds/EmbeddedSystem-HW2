CC = nvcc
OBJS = constant_mem.o user_device.o user_host.o user.o
CFLAGS = -c

all : constant_mem.out

constant_mem.out : $(OBJS)
	$(CC) -o constant_mem.out $(OBJS)

constant_mem.o : constant_mem.cu
	$(CC) $(CFLAGS) constant_mem.cu -o constant_mem.o

user_device.o : user_device.cu
	$(CC) $(CFLAGS) user_device.cu -o user_device.o

user_host.o : user_host.cu
	$(CC) $(CFLAGS) user_host.cu -o user_host.o

user.o : user.cu
	$(CC) $(CFLAGS) user.cu -o user.o

clean :
	rm -f constant_mem.out *.o

rebuild : clean all
