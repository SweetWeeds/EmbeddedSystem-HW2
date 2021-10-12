CC = nvcc
OBJS = maxFloatNum.o user_device.o user_host.o user.o
CFLAGS = -c

all : maxFloatNum.out

maxFloatNum.out : $(OBJS)
	$(CC) -o maxFloatNum.out $(OBJS)

maxFloatNum.o : maxFloatNum.cu
	$(CC) $(CFLAGS) -o maxFloatNum.o maxFloatNum.cu

user_device.o : user_device.cu
	$(CC) $(CFLAGS) -o user_device.o user_device.cu

user_host.o : user_host.cu
	$(CC) $(CFLAGS) -o user_host.o user_host.cu

user.o : user.cu
	$(CC) $(CFLAGS) -o user.o user.cu

clean :
	rm -f maxFloatNum.out *.o

rebuild : clean all
