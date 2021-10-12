CC = nvcc
OBJS = main.o user_device.o user_host.o csv_data.o user.o
CFLAGS = -c

all : maxFloatNum.out

maxFloatNum.out : $(OBJS)
	$(CC) -o maxFloatNum.out $(OBJS)

main.o : main.cu
	$(CC) $(CFLAGS) -o main.o main.cu

user_device.o : user_device.cu
	$(CC) $(CFLAGS) -o user_device.o user_device.cu

user_host.o : user_host.cu
	$(CC) $(CFLAGS) -o user_host.o user_host.cu

csv_data.o : csv_data.cu
	$(CC) $(CFLAGS) -o csv_data.o csv_data.cu

user.o : user.cu
	$(CC) $(CFLAGS) -o user.o user.cu

clean :
	rm -f maxFloatNum.out *.o

rebuild : clean all
