CC = nvcc
OBJS = disable_L1_Cache.o user_device.o user_host.o user.o
CFLAGS = -c -Xptxas -dlcm=cg

all : disable_L1_Cache.out

disable_L1_Cache.out : $(OBJS)
	$(CC) -o disable_L1_Cache.out $(OBJS)

disable_L1_Cache.o : disable_L1_Cache.cu
	$(CC) $(CFLAGS) disable_L1_Cache.cu -o disable_L1_Cache.o

user_device.o : user_device.cu
	$(CC) $(CFLAGS) user_device.cu -o user_device.o

user_host.o : user_host.cu
	$(CC) $(CFLAGS) user_host.cu -o user_host.o

user.o : user.cu
	$(CC) $(CFLAGS) user.cu -o user.o

clean :
	rm -f disable_L1_Cache.out *.o

rebuild : clean all
