CC = nvcc
OBJS = not_use_mod_with_L1_diable.o user_device.o user_host.o user.o
CFLAGS = -c -Xptxas -dlcm=cg

all : not_use_mod_with_L1_diable.out

not_use_mod_with_L1_diable.out : $(OBJS)
	$(CC) -o not_use_mod_with_L1_diable.out $(OBJS)

not_use_mod_with_L1_diable.o : not_use_mod_with_L1_diable.cu
	$(CC) $(CFLAGS) not_use_mod_with_L1_diable.cu -o not_use_mod_with_L1_diable.o

user_device.o : user_device.cu
	$(CC) $(CFLAGS) user_device.cu -o user_device.o

user_host.o : user_host.cu
	$(CC) $(CFLAGS) user_host.cu -o user_host.o

user.o : user.cu
	$(CC) $(CFLAGS) user.cu -o user.o

clean :
	rm -f not_use_mod_with_L1_diable.out *.o

rebuild : clean all
