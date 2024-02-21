# 编译器
CC := nvcc

# 编译选项
CFLAGS := -g -O3 -arch=sm_75 --ptxas-options=-v -lineinfo

src_file := $(bin_name).cu

include_dir := include

# 目标规则
all: $(bin_name)

# 生成二进制文件
$(bin_name): $(src_file)
	mkdir -p bin
	$(CC) $(CFLAGS) -I$(include_dir) -o bin/$@ $<

# 清理生成的文件
clean:
	rm -rf bin/$(bin_name)

.PHONY: all clean
