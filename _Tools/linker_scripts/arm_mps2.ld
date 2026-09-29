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

/* Linker script for ARM Cortex-M4 on QEMU MPS2-AN386
 * Memory map: Flash at 0x00000000 (4MB), SRAM at 0x20000000 (4MB)
 */

MEMORY
{
    FLASH (rx)  : ORIGIN = 0x00000000, LENGTH = 4M
    RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 4M
}

ENTRY(Reset_Handler)

_stack_top = ORIGIN(RAM) + LENGTH(RAM);

SECTIONS
{
    .isr_vector : {
        KEEP(*(.isr_vector))
        . = ALIGN(4);
    } > FLASH

    .text : {
        *(.text*)
        *(.rodata*)
        *(.ARM.extab*)
        *(.ARM.exidx*)
        . = ALIGN(4);
        _etext = .;
    } > FLASH

    .data : AT (_etext) {
        _sdata = .;
        *(.data*)
        . = ALIGN(4);
        _edata = .;
    } > RAM

    .bss : {
        _sbss = .;
        *(.bss*)
        *(COMMON)
        . = ALIGN(4);
        _ebss = .;
    } > RAM

    _end = .;
    PROVIDE(end = .);
    PROVIDE(_heap_start = .);
    _heap_end = ORIGIN(RAM) + LENGTH(RAM) - 1024;  /* Reserve 1KB for stack guard */

    /* Remove unwanted sections */
    /DISCARD/ : {
        *(.ARM.attributes)
        *(.comment)
        *(.eh_frame)
    }
}
