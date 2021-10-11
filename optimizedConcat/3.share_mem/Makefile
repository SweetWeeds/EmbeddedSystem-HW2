CC = nvcc
OBJS = share_mem.o user_device.o user_host.o user.o
CFLAGS = -c

all : share_mem.out

share_mem.out : $(OBJS)
	$(CC) -o share_mem.out $(OBJS)

share_mem.o : share_mem.cu
	$(CC) $(CFLAGS) share_mem.cu -o share_mem.o

user_device.o : user_device.cu
	$(CC) $(CFLAGS) user_device.cu -o user_device.o

user_host.o : user_host.cu
	$(CC) $(CFLAGS) user_host.cu -o user_host.o

user.o : user.cu
	$(CC) $(CFLAGS) user.cu -o user.o

clean :
	rm -f share_mem.out *.o

rebuild : clean all
