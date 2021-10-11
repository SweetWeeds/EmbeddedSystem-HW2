CC = nvcc
OBJS = not_use_L1.o user_device.o user_host.o user.o
CFLAGS = -c

all : not_use_L1.out

not_use_L1.out : $(OBJS)
	$(CC) -o not_use_L1.out $(OBJS)

not_use_L1.o : not_use_L1.cu
	$(CC) $(CFLAGS) not_use_L1.cu -o not_use_L1.o

user_device.o : user_device.cu
	$(CC) $(CFLAGS) user_device.cu -o user_device.o

user_host.o : user_host.cu
	$(CC) $(CFLAGS) user_host.cu -o user_host.o

user.o : user.cu
	$(CC) $(CFLAGS) user.cu -o user.o

clean :
	rm -f not_use_L1.out *.o

rebuild : clean all
