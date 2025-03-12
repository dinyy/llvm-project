.globl main
.type main, @function
main:
    addi    sp, sp, -8
    sd      ra, 0(sp)
    li      a0, 0
    ld      ra, 0(sp)
    addi    sp, sp, 8
    ret
.size main, .-main