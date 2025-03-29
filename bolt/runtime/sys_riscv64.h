#ifndef LLVM_TOOLS_LLVM_BOLT_SYS_RISCV64
#define LLVM_TOOLS_LLVM_BOLT_SYS_RISCV64

// Save all registers while keeping 16B stack alignment
#define SAVE_ALL                       \
"addi x2, x2, -256\n"                      \
"sd x0, 8(x2)\n"                           \
"sd x1, 16(x2)\n"                          \
"sd x3, 24(x2)\n"                          \
"sd x4, 32(x2)\n"                          \
"sd x5, 40(x2)\n"                          \
"sd x6, 48(x2)\n"                          \
"sd x7, 56(x2)\n"                          \
"sd x8, 64(x2)\n"                          \
"sd x9, 72(x2)\n"                          \
"sd x10, 80(x2)\n"                         \
"sd x11, 88(x2)\n"                         \
"sd x12, 96(x2)\n"                         \
"sd x13, 104(x2)\n"                        \
"sd x14, 112(x2)\n"                        \
"sd x15, 120(x2)\n"                        \
"sd x16, 128(x2)\n"                        \
"sd x17, 136(x2)\n"                        \
"sd x18, 144(x2)\n"                        \
"sd x19, 152(x2)\n"                        \
"sd x20, 160(x2)\n"                        \
"sd x21, 168(x2)\n"                        \
"sd x22, 176(x2)\n"                        \
"sd x23, 184(x2)\n"                        \
"sd x24, 192(x2)\n"                        \
"sd x25, 200(x2)\n"                        \
"sd x26, 208(x2)\n"                        \
"sd x27, 216(x2)\n"                        \
"sd x28, 224(x2)\n"                        \
"sd x29, 232(x2)\n"                        \
"sd x30, 240(x2)\n"                        \
"sd x31, 248(x2)\n"

// Mirrors SAVE_ALL
#define RESTORE_ALL                    \
"ld x31, 248(x2)\n"                        \
"ld x30, 240(x2)\n"                        \
"ld x29, 232(x2)\n"                        \
"ld x28, 224(x2)\n"                        \
"ld x27, 216(x2)\n"                        \
"ld x26, 208(x2)\n"                        \
"ld x25, 200(x2)\n"                        \
"ld x24, 192(x2)\n"                        \
"ld x23, 184(x2)\n"                        \
"ld x22, 176(x2)\n"                        \
"ld x21, 168(x2)\n"                        \
"ld x20, 160(x2)\n"                        \
"ld x19, 152(x2)\n"                        \
"ld x18, 144(x2)\n"                        \
"ld x17, 136(x2)\n"                        \
"ld x16, 128(x2)\n"                        \
"ld x15, 120(x2)\n"                        \
"ld x14, 112(x2)\n"                        \
"ld x13, 104(x2)\n"                        \
"ld x12, 96(x2)\n"                         \
"ld x11, 88(x2)\n"                         \
"ld x10, 80(x2)\n"                         \
"ld x9, 72(x2)\n"                          \
"ld x8, 64(x2)\n"                          \
"ld x7, 56(x2)\n"                          \
"ld x6, 48(x2)\n"                          \
"ld x5, 40(x2)\n"                          \
"ld x4, 32(x2)\n"                          \
"ld x3, 24(x2)\n"                          \
"ld x1, 16(x2)\n"                          \
"ld x0, 8(x2)\n"                           \
"addi x2, x2, 256\n"

// Anonymous namespace covering everything but our library entry point
namespace {


// Get the difference between runtime address of .text section and static
// address in section header table. Can be extracted from arbitrary pc value
// recorded at runtime to get the corresponding static address, which in turn
// can be used to search for indirect call description. Needed because indirect
// call descriptions are read-only non-relocatable data.
uint64_t getTextBaseAddress() {
  uint64_t DynAddr;
  uint64_t StaticAddr;
  __asm__ volatile("lla %0, __hot_end\n\t"
                   "lui %1, %%hi(__hot_end)\n\t"
                   "addi %1, %1, %%lo(__hot_end)\n\t"
                   : "=r"(DynAddr), "=r"(StaticAddr));
  return DynAddr - StaticAddr;
}


uint64_t __read(uint64_t fd, const void *buf, uint64_t count) {
  uint64_t ret;
  register uint64_t x10 __asm__("x10") = fd;
  register const void *x11 __asm__("x11") = buf;
  register uint64_t x12 __asm__("x12") = count;
  register uint64_t x17 __asm__("x17") = 63;
  __asm__ __volatile__("ecall\n"
                   "mv %0, x10"
                   : "=r"(ret),"+r"(x10), "+r"(x11)
                   : "r"(x12), "r"(x17)
                   : "memory");
  return ret;
}

uint64_t __write(uint64_t fd, const void *buf, uint64_t count) {
  uint64_t ret;
  register uint64_t x10 __asm__("x10") = fd;
  register const void *x11 __asm__("x11") = buf;
  register uint64_t x12 __asm__("x12") = count;
  register uint64_t x17 __asm__("x17") = 64;
  __asm__ __volatile__("ecall\n"
                   "mv %0, x10"
                   : "=r"(ret),"+r"(x10),"+r"(x11)
                   : "r"(x12), "r"(x17)
                   : "memory");
  return ret;
}

void *__mmap(uint64_t addr, uint64_t size, uint64_t prot, uint64_t flags,
             uint64_t fd, uint64_t offset) {
  void *ret;
  register uint64_t x10 __asm__("x10") = addr;
  register uint64_t x11 __asm__("x11") = size;
  register uint64_t x12 __asm__("x12") = prot;
  register uint64_t x13 __asm__("x13") = flags;
  register uint64_t x14 __asm__("x14") = fd;
  register uint64_t x15 __asm__("x15") = offset;
  register uint64_t x17 __asm__("x17") = 222;
  __asm__ __volatile__("ecall\n"
                   "mv %0, x10"
                   : "=r"(ret),"+r"(x10),"+r"(x11)
                   : "r"(x12), "r"(x13),"r"(x14), "r"(x15), "r"(x17)
                   : "memory");
  return ret;
}

uint64_t __munmap(void *addr, uint64_t size) {
  uint64_t ret;
  register void *x10 __asm__("x10") = addr;
  register uint64_t x11 __asm__("x11") = size;
  register uint64_t x17 __asm__("x17") = 215;
  __asm__ __volatile__("ecall\n"
                  "mv %0, x10"
                  : "=r"(ret),"+r"(x10),"+r"(x11)
                  : "r"(x17)
                  : "memory");
  return ret;
}

uint64_t __exit(uint64_t code) {
  uint64_t ret;
  register uint64_t x10 __asm__("x10") = code;
  register uint64_t x17 __asm__("x17") = 94;
  __asm__ __volatile__("ecall\n"
                   "mv %0, x10"
                   : "=r"(ret),"+r"(x10)
                   : "r"(x17)
                   : "memory","x11");
  return ret;
}

uint64_t __open(const char *pathname, uint64_t flags, uint64_t mode) {
  uint64_t ret;
  register int x10 __asm__("x10") = -100;
  register const char *x11 __asm__("x11") = pathname;
  register uint64_t x12 __asm__("x12") = flags;
  register uint64_t x13 __asm__("x13") = mode;
  register uint64_t x17 __asm__("x17") = 56;
  __asm__ __volatile__("ecall\n"
                   "mv %0, x10"
                   : "=r"(ret),"+r"(x10),"+r"(x11)
                   : "r"(x12), "r"(x13), "r"(x17)
                   : "memory");
  return ret;
}

long __getdents64(unsigned int fd, dirent64 *dirp, size_t count) {
  long ret;
  register unsigned int x10 __asm__("x10") = fd;
  register dirent64 *x11 __asm__("x11") = dirp;
  register size_t x12 __asm__("x12") = count;
  register uint64_t x17 __asm__("x17") = 61;
  __asm__ __volatile__("ecall\n"
                   "mv %0, x10"
                   : "=r"(ret),"+r"(x10),"+r"(x11)
                   : "r"(x12), "r"(x17)
                   : "memory");
  return ret;
}

uint64_t __readlink(const char *pathname, char *buf, size_t bufsize) {
  uint64_t ret;
  register int x10 __asm__("x10") = -100;
  register const char *x11 __asm__("x11") = pathname;
  register char *x12 __asm__("x12") = buf;
  register size_t x13 __asm__("x13") = bufsize;
  register uint64_t x17 __asm__("x17") = 78; // readlinkat
  __asm__ __volatile__("ecall\n"
                   "mv %0, x10"
                  : "=r"(ret),"+r"(x10),"+r"(x11)
                  : "r"(x12), "r"(x13),"r"(x17)
                  : "memory");
  return ret;
}

uint64_t __lseek(uint64_t fd, uint64_t pos, uint64_t whence) {
  uint64_t ret;
  register uint64_t x10 __asm__("x10") = fd;
  register uint64_t x11 __asm__("x11") = pos;
  register uint64_t x12 __asm__("x12") = whence;
  register uint64_t x17 __asm__("x17") = 62;
  __asm__ __volatile__("ecall\n"
                   "mv %0, x10"
                   : "=r"(ret), "+r"(x10),"+r"(x11)
                   : "r"(x12), "r"(x17)
                   : "memory");
  return ret;
}

int __ftruncate(uint64_t fd, uint64_t length) {
  int ret;
  register uint64_t x10 __asm__("x10") = fd;
  register uint64_t x11 __asm__("x11") = length;
  register uint64_t x17 __asm__("x17") = 46;
  __asm__ __volatile__("ecall\n"
                   "mv %0, x10"
                   : "=r"(ret), "+r"(x10),"+r"(x11)
                   : "r"(x17)
                   : "memory");
  return ret;
}

int __close(uint64_t fd) {
  int ret;
  register uint64_t x10 __asm__("x10") = fd;
  register uint64_t x17 __asm__("x17") = 57;
  __asm__ __volatile__("ecall\n"
                   "mv %0, x10"
                   : "=r"(ret), "+r"(x10)
                   : "r"(x17)
                   : "memory","x11");
  return ret;
}

int __madvise(void *addr, size_t length, int advice) {
  int ret;
  register void *x10 __asm__("x10") = addr;
  register size_t x11 __asm__("x11") = length;
  register int x12 __asm__("x12") = advice;
  register uint64_t x17 __asm__("x17") = 233;
  __asm__ __volatile__("ecall\n"
                  "mv %0, x10"
                  : "=r"(ret), "+r"(x10), "+r"(x11)
                  : "r"(x12),"r"(x17)
                  : "memory");
  return ret;
}

int __uname(struct UtsNameTy *buf) {
  int ret;
  register UtsNameTy *x10 __asm__("x10") = buf;
  register uint64_t x17 __asm__("x17") = 160;
  __asm__ __volatile__("ecall\n"
                  "mv %0, x10"
                  : "=r"(ret), "+r"(x10)
                  : "r"(x17)
                  : "memory","x11");
  return ret;
}

uint64_t __nanosleep(const timespec *req, timespec *rem) {
  uint64_t ret;
  register const timespec *x10 __asm__("x10") = req;
  register timespec *x11 __asm__("x11") = rem;
  register uint64_t x17 __asm__("x17") = 101;
  __asm__ __volatile__("ecall\n"
                  "mv %0, x10"
                  : "=r"(ret), "+r"(x10), "+r"(x11)
                  : "r"(x17)
                  : "memory");
  return ret;
}

int64_t __fork() {
  uint64_t ret;
  // clone instead of fork with flags
  // "CLONE_CHILD_CLEARTID|CLONE_CHILD_SETTID|SIGCHLD"
  register uint64_t x10 __asm__("x10") = 0x1200011;
  register uint64_t x11 __asm__("x11") = 0;
  register uint64_t x12 __asm__("x12") = 0;
  register uint64_t x13 __asm__("x13") = 0;
  register uint64_t x14 __asm__("x14") = 0;
  register uint64_t x17 __asm__("x17") = 220;
  __asm__ __volatile__("ecall\n"
                  "mv %0, x10"
                  : "=r"(ret), "+r"(x10), "+r"(x11)
                  : "r"(x12),"r"(x13),"r"(x14),"r"(x17)
                  : "memory");
  return ret;
}

int __mprotect(void *addr, size_t len, int prot) {
  int ret;
  register void *x10 __asm__("x10") = addr;
  register size_t x11 __asm__("x11") = len;
  register int x12 __asm__("x12") = prot;
  register uint64_t x17 __asm__("x17") = 226;
  __asm__ __volatile__("ecall\n"
                  "mv %0, x10"
                  : "=r"(ret), "+r"(x10), "+r"(x11)
                  : "r"(x12),"r"(x17)
                  : "memory");
  return ret;
}

uint64_t __getpid() {
  uint64_t ret;
  register uint64_t x17 __asm__("x17") = 172;
  __asm__ __volatile__("ecall\n"
                  "mv %0, x10"
                  : "=r"(ret) 
                  : "r"(x17)
                  : "memory","x10","x11");
  return ret;
}

uint64_t __getppid() {
  uint64_t ret;
  register uint64_t x17 __asm__("x17") = 173;
  __asm__ __volatile__("ecall\n"
                  "mv %0, x10"
                  : "=r"(ret) 
                  : "r"(x17)
                  : "memory","x10","x11");
  return ret;
}

int __setpgid(uint64_t pid, uint64_t pgid) {
  int ret;
  register uint64_t x10 __asm__("x10") = pid;
  register uint64_t x11 __asm__("x11") = pgid;
  register uint32_t x17 __asm__("x17") = 154;
  __asm__ __volatile__("ecall\n"
                  "mv %0, x10"
                  : "=r"(ret) ,"+r"(x10),"+r"(x11)
                  : "r"(x17)
                  : "memory");
  return ret;
}

uint64_t __getpgid(uint64_t pid) {
  uint64_t ret;
  register uint64_t x10 __asm__("x10") = pid;
  register uint64_t x17 __asm__("x17") = 155;
  __asm__ __volatile__("ecall\n"
                  "mv %0, x10"
                  : "=r"(ret) ,"+r"(x10)
                  : "r"(x17)
                  : "memory","x11");
  return ret;
}

int __kill(uint64_t pid, int sig) {
  int ret;
  register uint64_t x10 __asm__("x10") = pid;
  register int x11 __asm__("x11") = sig;
  register uint64_t x17 __asm__("x17") = 129;
  __asm__ __volatile__("ecall\n"
                  "mv %0, x10"
                  : "=r"(ret) ,"+r"(x10),"+r"(x11)
                  : "r"(x17)
                  : "memory");
  return ret;
}

int __fsync(int fd) {
  int ret;
  register int x10 __asm__("x10") = fd;
  register uint64_t x17 __asm__("x17") = 82;
  __asm__ __volatile__("ecall\n"
                  "mv %0, x10"
                  : "=r"(ret) 
                  : "r"(x17)
                  : "memory","x10","x11");
  return ret;
}

uint64_t __sigprocmask(int how, const void *set, void *oldset) {
  uint64_t ret;
  register int x10 __asm__("x10") = how;
  register const void *x11 __asm__("x11") = set;
  register void *x12 __asm__("x12") = oldset;
  register long x13 asm("x13") = 8;
  register uint64_t x17 __asm__("x17") = 135;
  __asm__ __volatile__("ecall\n"
                  "mv %0, x10"
                  : "=r"(ret) ,"+r"(x10) ,"+r"(x11)
                  : "r"(x12) ,"r"(x13) ,"r"(x17)
                  : "memory");
  return ret;
}

int __prctl(int option, unsigned long arg2, unsigned long arg3,
            unsigned long arg4, unsigned long arg5) {
  int ret;
  register int x10 __asm__("x10") = option;
  register unsigned long x11 __asm__("x11") = arg2;
  register unsigned long x12 __asm__("x12") = arg3;
  register unsigned long x13 __asm__("x13") = arg4;
  register unsigned long x14 __asm__("x14") = arg5;
  register uint64_t x17 __asm__("x17") = 167;
  __asm__ __volatile__("ecall\n"
                  "mv %0, x10"
                  : "=r"(ret) ,"+r"(x10),"+r"(x11)
                  : "r"(x12) ,"r"(x13) ,"r"(x14) ,"r"(x17)
                  : "memory");
  return ret;
}


} // anonymous namespace

#endif