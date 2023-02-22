# 訓練 ChatGPT

前情提要: 原本我問 ChatGPT 有關 RISC-V 的組合語言，他幾乎都不會，請看 [12-asm](../12-asm)

然後我把自己寫的這篇文章貼進去

* https://github.com/riscv2os/riscv2os/wiki/eosHelloOs


結果 ChatGPT 回應了下列這段話:

> 在本篇文章中，作者將介紹如何撰寫一個簡單的 RISC-V 處理器上的嵌入式作業系統，稱之為 mini-arm-os。作者將會透過一系列的程式碼來說明如何實作嵌入式作業系統。
> 
> 作者提到，作者所使用的 RISC-V 虛擬機在 QEMU 中稱為 virt，並且指出 UART 記憶體映射位置從 0x10000000 開始，並且提供了 UART 映射區的記憶體位置。作者也寫了一個函式 lib_putc，該函式可以將一個字元送到 UART 上，並且透過宿主機顯示在螢幕上。


接著從我和 ChatGPT 的問答中，你可以看到他學會了 ...

## ccc: 請問 RISC-V 虛擬機 virt 的 UART 記憶體映射位置從哪開始? 

RISC-V 虛擬機 virt 的 UART 記憶體映射位置從 0x10000000 開始。

## ccc: 映射位址 0x10000003 裡面有甚麼?

在 RISC-V 虛擬機 virt 中，UART 記憶體映射位置從 0x10000000 開始。因此，映射位址 0x10000003 是 UART 的 LCR (Line Control Register)。

LCR 用來控制 UART 的傳輸模式，包括傳輸速率、字元長度、奇偶校驗、停止位數等。具體內容見下表：

```
0x10000003 LCR (Line Control Register)

Bit 7 Bit 6 Bit 5 Bit 4 Bit 3 Bit 2 Bit 1 Bit 0

DLAB BC - EPS PEN STB WLS1 WLS0

DLAB: Divisor Latch Access Bit
BC: Break Control
EPS: Even Parity Select
PEN: Parity Enable
STB: Stop Bits
WLS1: Word Length Select Bit 1
WLS0: Word Length Select Bit 0
```

其中 DLAB 要求為 1 才能讀取和寫入 DLR (Divisor Latch Register)，以控制傳輸速率。

例如，如果我們想將 UART 的傳輸速率設為 9600，則可以寫入下列值：

```
*UART_LCR = 0x83; // Set DLAB=1
*UART_DLL = 0x38; // Baud rate = 9600
*UART_DLM = 0x00;
*UART_LCR = 0x03; // Set DLAB=0
```

注意，上面的設定並不是標準的方式，而是為了方便演示，实际使用時需要注意具體的


## ccc: 如何在 RISC-V 中印出一個字串

您可以使用 lib_puts() 函式來印出一個字串。該函式會逐個字元呼叫 lib_putc() 函式將字元印出。

範例程式碼：

```c
void lib_puts(char *s) {
    while (*s) lib_putc(*s++);
}

int os_main(void)
{
    lib_puts("Hello OS!\n");
    while (1) {}
    return 0;
}
```

在這個程式碼中，lib_puts() 函式會從給定的字串 *s 開始，一個字元一個字元地呼叫 lib_putc() 函式來將字元印出。

## ccc: 要怎麼用 QEMU 跑 RISC-V 的程式?

要在 QEMU 中跑 RISC-V 的程式，需要先安裝好 QEMU 並安裝 RISC-V 的模擬器。安裝方式可以參考 QEMU 的官方文件。

安裝完成後，可以使用以下指令來執行 RISC-V 程式：

```
$ qemu-system-riscv32 -machine virt -kernel [kernel image]
```

其中，[kernel image] 是指 RISC-V 的核心映像檔 (Kernel image)，通常會用二進制檔案 (ELF 檔)。

例如，如果想要執行一個名為 hello.elf 的 RISC-V 程式，可以使用以下指令：

```
$ qemu-system-riscv32 -machine virt -kernel hello.elf
```

執行完成後，QEMU 會啟動 RISC-V 的模擬器，並執行指定的程式。

## ccc: 請問如何用 RISC-V 印出 How are you ?

RISC-V 語言沒有特定的函數來印出 "How are you?"，但可以透過自行寫函數來完成這項任務。

例如，可以寫一個名為 lib_puts 的函數，接收一個字串參數，然後將該字串的每個字元逐一印出。在 RISC-V 中，這個函數可能會長得像這樣：

```c
void lib_puts(char *s) {
    // 將 s 的每個字元逐一印出
    while (*s) lib_putc(*s++);
}
// lib_putc 的程式碼請見上面的解答
int lib_putc(char ch) {
// 確認 UART 傳送區是否為空，如果不是則等待
while ((*UART_LSR & UART_LSR_EMPTY_MASK) == 0);
// 將 ch 寫入 UART 傳送區
return *UART_THR = ch;
}

// os_main 的程式碼請見上面的解答
int os_main(void) {
// 印出 "How are you?"
lib_puts("How are you?\n");
// 無限迴圈，防止程式結束
while (1) {}

return 0;
}
```

在呼叫 lib_puts 時，傳入 "How are you?" 字串，即可將 "How are you?" 印出。

請注意，本範例

