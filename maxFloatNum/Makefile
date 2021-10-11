CC = nvcc
OBJS = maxFloatNum.o user_device.o user_host.o user.o
CFLAGS = -c

all : maxFloatNum.out

maxFloatNum.out : $(OBJS)
	$(CC) -o maxFloatNum.out $(OBJS)

maxFloatNum.o : maxFloatNum.cu
	$(CC) $(CFLAGS) maxFloatNum.cu -o maxFloatNum.o

user_device.o : user_device.cu
	$(CC) $(CFLAGS) user_device.cu -o user_device.o

user_host.o : user_host.cu
	$(CC) $(CFLAGS) user_host.cu -o user_host.o

user.o : user.cu
	$(CC) $(CFLAGS) user.cu -o user.o

clean :
	rm -f maxFloatNum.out *.o

rebuild : clean all
