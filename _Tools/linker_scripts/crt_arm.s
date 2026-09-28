/*
 * Copyright (c) 2025, Infineon Technologies AG, or an affiliate of Infineon Technologies AG. All rights reserved.
 * This software, associated documentation and materials ("Software") is owned by Infineon Technologies AG or one
 * of its affiliates ("Infineon") and is protected by and subject to worldwide patent protection, worldwide copyright laws,
 * and international treaty provisions. Therefore, you may use this Software only as provided in the license agreement accompanying
 * the software package from which you obtained this Software. If no license agreement applies, then any use, reproduction, modification,
 * translation, or compilation of this Software is prohibited without the express written permission of Infineon.
 * Disclaimer: UNLESS OTHERWISE EXPRESSLY AGREED WITH INFINEON, THIS SOFTWARE IS PROVIDED AS-IS, WITH NO WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, ALL WARRANTIES OF NON-INFRINGEMENT OF THIRD-PARTY RIGHTS AND IMPLIED WARRANTIES
 * SUCH AS WARRANTIES OF FITNESS FOR A SPECIFIC USE/PURPOSE OR MERCHANTABILITY. Infineon reserves the right to make changes to the Software
 * without notice. You are responsible for properly designing, programming, and testing the functionality and safety of your intended application
 * of the Software, as well as complying with any legal requirements related to its use. Infineon does not guarantee that the Software will be
 * free from intrusion, data theft or loss, or other breaches ("Security Breaches"), and Infineon shall have no liability arising out of any
 * Security Breaches. Unless otherwise explicitly approved by Infineon, the Software may not be used in any application where a failure of the
 * Product or any consequences of the use thereof can reasonably be expected to result in personal injury.
*/

/* Minimal Cortex-M startup for QEMU semihosting
 * Provides vector table, Reset_Handler, and data/bss init.
 */

    .syntax unified
    .cpu cortex-m4
    .thumb

    .section .isr_vector, "a", %progbits
    .align 2
    .globl __isr_vector
__isr_vector:
    .word _stack_top        /* Initial Stack Pointer */
    .word Reset_Handler     /* Reset Handler */
    .word Default_Handler   /* NMI */
    .word Default_Handler   /* HardFault */
    .word Default_Handler   /* MemManage */
    .word Default_Handler   /* BusFault */
    .word Default_Handler   /* UsageFault */
    .word 0, 0, 0, 0       /* Reserved */
    .word Default_Handler   /* SVCall */
    .word Default_Handler   /* Debug Monitor */
    .word 0                 /* Reserved */
    .word Default_Handler   /* PendSV */
    .word Default_Handler   /* SysTick */

    .section .text
    .align 2
    .thumb_func
    .globl Reset_Handler
    .type Reset_Handler, %function
Reset_Handler:
    /* Enable FPU (CP10/CP11 full access) */
    ldr r0, =0xE000ED88     /* CPACR */
    ldr r1, [r0]
    orr r1, r1, #(0xF << 20)  /* CP10 + CP11 full access */
    str r1, [r0]
    dsb
    isb

    /* Copy .data from flash to RAM */
    ldr r0, =_sdata
    ldr r1, =_edata
    ldr r2, =_etext
.Lcopy_data:
    cmp r0, r1
    bge .Lzero_bss
    ldr r3, [r2], #4
    str r3, [r0], #4
    b .Lcopy_data

    /* Zero .bss */
.Lzero_bss:
    ldr r0, =_sbss
    ldr r1, =_ebss
    movs r2, #0
.Lzero_loop:
    cmp r0, r1
    bge .Lcall_main
    str r2, [r0], #4
    b .Lzero_loop

.Lcall_main:
    /* Initialize semihosting stdio (rdimon) */
    bl initialise_monitor_handles
    /* Call main */
    bl main

    /* Exit via semihosting SYS_EXIT (0x18) */
    movs r0, #0x18     /* SYS_EXIT */
    ldr r1, =.Lexit_args
    bkpt #0xAB         /* Semihosting breakpoint */
    b .                 /* Should not reach here */

    .align 2
.Lexit_args:
    .word 0x20026       /* ADP_Stopped_ApplicationExit */
    .word 0             /* Exit code 0 */

    .thumb_func
    .globl Default_Handler
    .type Default_Handler, %function
Default_Handler:
    b .                 /* Infinite loop */

    .end
