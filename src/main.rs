use core::panic;
use rand::Rng;
use std::{env, fs::File,  io::{ BufRead, BufReader, Error, Read}};
use minifb::{Key, Window, WindowOptions};
use std::time::{Duration, Instant};



const DISPLAY_HEIGHT: usize = 32;
const DISPLAY_WIDTH: usize = 64;
const WINDOW_SCALE: usize = 10;

const VF: usize = 0xF;

struct Chip8 {
    memory: [u8; 4096],
    registers: [u8; 16],
    index_register: u16,
    program_counter: u16,
    stack: Vec<u16>,
    stack_pointer: usize,
    display_buffer: [[bool; DISPLAY_WIDTH]; DISPLAY_HEIGHT],
    keypad: [bool; 16],
    delay_timer: u8,
    sound_timer: u8,
    keypad_pressed: Option<usize>,
}

impl Chip8 {
    fn new() -> Self {
        // Initialize CPU state
        Chip8 {
            memory: [0; 4096],
            registers: [0; 16],
            index_register: 0,
            program_counter: 0x200, // Start execution at address 0x200
            stack: Vec::new(),
            stack_pointer: 0,
            display_buffer: [[false; DISPLAY_WIDTH]; DISPLAY_HEIGHT],
            keypad: [false; 16],
            delay_timer: 0,
            sound_timer: 0,
            keypad_pressed: None,
        }
    }
    
    // TODO Cannot read ch8 roms right now or reads the 00000 at the begining
    pub fn load_rom(&mut self, file_path: &str) -> Result<(), Error> {
        // Open the file
        let mut file = File::open(file_path)?;

        // Get the size of the file
        let file_size = file.metadata()?.len() as usize;

        // Read the entire file into a buffer
        let mut buffer = vec![0u8; file_size];
        file.read_exact(&mut buffer)?;

        // Load the ROM into memory starting at address 0x200
        let rom_start_address = 0x200;
        self.memory[rom_start_address..(rom_start_address + file_size)].copy_from_slice(&buffer);

        Ok(())
    }

/* 
pub fn load_rom(&mut self, file_path: &str) -> Result<(), Error> {
    // Open the file
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    // Load the ROM into memory starting at address 0x200
    let mut address = 0x200;
    for line_result in reader.lines() {
        let line = line_result?;
        let bytes: Vec<u8> = line
            .split_whitespace()
            .skip_while(|&byte| byte.chars().next().unwrap_or('0').is_digit(10))
            .filter_map(|byte| u8::from_str_radix(byte, 16).ok())
            .collect();

        // Load the bytes into memory
        for byte in bytes {
            self.memory[address] = byte;
            address += 1;
        }
    }

    Ok(())
}
*/

    pub fn fetch_opcode(&self) -> u16 {
        let pc = self.program_counter as usize;
        let opcode = u16::from(self.memory[pc]) << 8 | u16::from(self.memory[pc + 1]);
        opcode
    }

    fn draw_sprite(&mut self, vx: usize, vy: usize, height: usize) {
        let x = self.registers[vx] as usize;
        let y = self.registers[vy] as usize;

        self.registers[0xF] = 0; // Reset collision flag

        for row in 0..height {
            let sprite_row = self.memory[self.index_register as usize + row];
            for col in 0..8 {
                let pixel_value = (sprite_row >> (7 - col)) & 0x1;
                let x_pos = (x + col) % DISPLAY_WIDTH;
                let y_pos = (y + row) % DISPLAY_HEIGHT;
                let current_pixel = self.display_buffer[y_pos][x_pos];
                if pixel_value == 1 && current_pixel {
                    self.registers[0xF] = 1; // Set collision flag
                }
                self.display_buffer[y_pos][x_pos] ^= pixel_value == 1;
            }
        }
    }

    // Helper function to handle the SKP Vx instruction
    fn skip_if_key_pressed(&mut self, vx: usize) {
        let key = self.registers[vx] as usize;
        if self.keypad[key] {
            self.program_counter += 2; // Skip the next instruction
        }
    }

    fn skip_if_key_not_pressed(&mut self, vx: usize) {
        let key = self.registers[vx] as usize;
        if !self.keypad[key] {
            self.program_counter += 2; // Skip the next instruction
        }
    }

    // It should clear the display turning all pixels off to 0
    fn clear_screen(&mut self) {
        self.display_buffer = [[false; DISPLAY_WIDTH]; DISPLAY_HEIGHT];
    }

    fn jump(&mut self, addr: u16) {
        self.program_counter = addr;
    }

    fn call(&mut self, addr: u16) {
        self.stack.push(self.program_counter);
        self.program_counter = addr;
    }

    fn skip_if_equal(&mut self, register: usize, value: u8) {
        if self.registers[register] == value {
            self.program_counter += 2;
        }
    }

    fn skip_if_not_equal(&mut self, register: usize, value: u8) {
        if self.registers[register] != value {
            self.program_counter += 2;
        }
    }

    fn skip_if_equal_registers(&mut self, register_x: usize, register_y: usize) {
        if self.registers[register_x] == self.registers[register_y] {
            self.program_counter += 2;
        }
    }

    // If the result is greater than 8 bits (i.e., > 255,) VF is set to 1, otherwise 0
    fn add(&mut self, register_x: usize, register_y: usize) {
        let (result, overlfow) =
            self.registers[register_x].overflowing_add(self.registers[register_y]);
        self.registers[register_x] = result;
        self.registers[VF] = if overlfow { 1 } else { 0 };
    }

    // If Vx > Vy, then VF is set to 1, otherwise 0.
    fn sub(&mut self, register_x: usize, register_y: usize) {
        if self.registers[register_x] > self.registers[register_y] {
            self.registers[VF] = 1;
        } else {
            self.registers[VF] = 0;
        }
        // TODO check overflow
        // self.registers[register_x] -= self.registers[register_y];
        self.registers[register_x] =
            self.registers[register_x].wrapping_sub(self.registers[register_y]);
    }

    fn shr(&mut self, register_x: usize) {
        self.registers[VF] = self.registers[register_x] & 0x1;
        self.registers[register_x] >>= 1;
    }

    fn subn(&mut self, register_x: usize, register_y: usize) {
        if self.registers[register_y] > self.registers[register_x] {
            self.registers[VF] = 1;
        } else {
            self.registers[VF] = 0;
        }

        self.registers[register_x] = self.registers[register_y] - self.registers[register_x];
    }

    fn shl(&mut self, register_x: usize) {
        self.registers[VF] = (self.registers[register_x] & 0x80) >> 7;
        self.registers[register_x] <<= 1;
    }

    fn skip_if_not_equal_registers(&mut self, register_x: usize, register_y: usize) {
        if self.registers[register_x] != self.registers[register_y] {
            self.program_counter += 2;
        }
    }

    fn random_byte(&mut self, register_x: usize, byte: u8) {
        let mut rng = rand::thread_rng();
        let random_byte: u8 = rng.gen();
        self.registers[register_x] = random_byte & byte;
    }

    fn wait_key_pressed(&mut self, register_x: usize) {
        // Reset the flag indicating a key press
        self.keypad_pressed = None;

        // Update the register with the pressed key, if any
        while self.keypad_pressed.is_none() {
            // Check for key press
            for (i, &key_state) in self.keypad.iter().enumerate() {
                if key_state {
                    // Key is pressed
                    self.registers[register_x] = i as u8;
                    self.keypad_pressed = Some(i);
                    break;
                }
            }

            // Sleep for a short duration to avoid busy-waiting
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    fn set_index_to_sprite(&mut self, register_x: usize) {
        let character = self.registers[register_x] as u16;
        self.index_register = character * 5; // each character is 5 bytes
    }

    fn store_bcd(&mut self, register_x: usize) {
        let value = self.registers[register_x];
        self.memory[self.index_register as usize] = value / 100;
        self.memory[(self.index_register + 1) as usize] = (value / 10) % 10; // Tens digit
        self.memory[(self.index_register + 2) as usize] = value % 10; // Ones digit
    }

    fn store_registers(&mut self, register_x: usize) {
        for i in 0..=register_x {
            self.memory[(self.index_register + i as u16) as usize] = self.registers[i];
        }
    }

    fn load_registers(&mut self, register_x: usize) {
        for i in 0..=register_x {
            self.registers[i] = self.memory[(self.index_register + i as u16) as usize];
        }
    }

    // Translate it to an instruction
    fn execute_opcode(&mut self, opcode: u16) {
        // Decode and execute opcode
        match opcode {
            // CLS
            // Clear the display
            0x00E0 => self.clear_screen(),

            // RET
            // The interpreter sets the program counter to the address at the top of the stack
            // then subtracts 1 from the stack pointer.
            0x00EE => {
                self.program_counter = self.stack[self.stack_pointer];
                self.stack[self.stack_pointer] = 0;
            }

            //JP addr
            // Jump to location nnn
            0x1000..=0x1FFF => {
                let addr = opcode & 0x0FFF;
                self.jump(addr);
            }

            // CALL addr
            // Call subroutine at nnn
            0x2000..=0x2FFF => {
                let addr = opcode & 0x0FFF;
                self.call(addr);
            }

            // SE vx, byte
            // Skip next instruction if Vx = kk
            0x3000..=0x3FFF => {
                let vx = ((opcode & 0x0F00) >> 8) as usize;
                let kk = (opcode & 0x00FF) as u8;
                self.skip_if_equal(vx, kk);
            }

            // SNE Vx, byte
            // Skip next instruction if Vx != kk
            0x4000..=0x4FFF => {
                let vx = ((opcode & 0x0F00) >> 8) as usize;
                let kk = (opcode & 0x00FF) as u8;
                self.skip_if_not_equal(vx, kk);
            }

            // SE Vx, Vy
            // Skip next instruction if Vx = Vy
            0x5000..=0x5FFF => {
                let vx = ((opcode & 0x0F00) >> 8) as usize;
                let vy = ((opcode & 0x00F0) >> 4) as usize; // TODO whoudl be << 4 or 8?
                self.skip_if_equal_registers(vx, vy);
            }

            // LD Vx, byte
            // Set Vx = kk
            0x6000..=0x6FFF => {
                let vx = ((opcode & 0x0F00) >> 8) as usize;
                let kk = (opcode & 0x00FF) as u8;
                self.registers[vx] = kk;
            }

            // ADD vx, byte
            // Sets Vx =  Vx + kk
            0x700..=0x7FFF => {
                let vx = ((opcode & 0x0F00) >> 8) as usize;
                let kk = (opcode & 0x00FF) as u8;
                //self.registers[vx] += kk;
                self.registers[vx] = self.registers[vx].wrapping_add(kk);
            }

            0x8000..=0x8FFF => {
                let vx = ((opcode & 0x0F00) >> 8) as usize;
                let vy = ((opcode & 0x00F0) >> 4) as usize;

                match opcode & 0x000F {
                    // LD Vx, Vy
                    // Set Vx = Vy
                    0x0 => self.registers[vx] = vy as u8,

                    // OR Vx, Vy
                    // Set Vx = Vx OR Vy
                    0x1 => self.registers[vx] |= self.registers[vy],

                    // AND Vx, Vy
                    // set Vx = Vx AND Vy
                    0x2 => self.registers[vx] &= self.registers[vy],

                    // XOR Vx, Vy
                    // Set Vx = Vx OR Vy
                    0x3 => self.registers[vx] ^= self.registers[vy],

                    // ADD Vx, Vy
                    // Set Vx = Vx + Vy, set VF = carry
                    0x4 => self.add(vx, vy),

                    // SUB Vx, Vy
                    // Set Vx = Vx - Vy, set VF = Not borrow
                    0x5 => self.sub(vx, vy),

                    // SHR Vx {, Vy}
                    // Set Vx = Vx SHR 1
                    0x6 => self.shr(vx),

                    // SUBN Vx, Vy
                    // Set Vx = Vy - Vx, set VF = not borrow
                    0x7 => self.subn(vx, vy),

                    // SHL Vx {, Vy}
                    0xE => self.shl(vx),

                    _ => panic!("Not a valid operation!"),
                }
            }

            // SNE Vx, Vy
            // Skip instruction if Vx != Vy
            0x9000..=0x9FF0 => {
                let vx = ((opcode & 0x0F00) >> 8) as usize;
                let vy = ((opcode & 0x00F0) >> 4) as usize;
                self.skip_if_not_equal_registers(vx, vy);
            }

            // LD I, addr
            // Set I = nnn
            0xA000..=0xAFFF => {
                let addr: u16 = opcode & 0x0FFF;
                self.index_register = addr;
            }

            // JP v0, addr
            // Jump to location addr + V0
            0xB000..=0xBFFF => {
                let addr = opcode & 0x0FFF;
                self.jump(addr + self.registers[0] as u16);
            }

            // RND Vx, byte
            // Set Vx = random byte AND kk
            0xC000..=0xCFFF => {
                let vx = ((opcode & 0x0F00) >> 8) as usize;
                let kk = (opcode & 0x00FF) as u8;
                self.random_byte(vx, kk);
            }

            // DRW Vx, Vy, nibble
            // Display n-byte starting at memory location I at (Vx, Vy), set VF = collision
            0xD000..=0xDFFF => {
                let vx = ((opcode & 0x0F00) >> 8) as usize;
                let vy = ((opcode & 0x00F0) >> 4) as usize;
                let height = (opcode & 0x000F) as usize;
                self.draw_sprite(vx, vy, height);
            }

            // SKP Vx && SKPN Vx
            // Skip next instruction if key with value of Vx is pressed (or not pressed for SKPN)
            0xE09E..=0xEFFF => {
                let vx = ((opcode & 0x0F00) >> 8) as usize;
                let last_nibble = opcode & 0x00FF;
                if last_nibble == 0x9E {
                    self.skip_if_key_pressed(vx);
                } else if last_nibble == 0xA1 {
                    self.skip_if_key_not_pressed(vx);
                } else {
                    unimplemented!("Unknown opcode: {:04X}", opcode);
                }
            }

            0xF000..=0xFFFF => {
                let vx = ((opcode & 0x0F00) >> 8) as usize;
                let last_nibble = opcode & 0x00FF;

                match last_nibble {
                    //LD Vx, DT
                    // Set Vx =  delay timer value
                    0x07 => self.registers[vx] = last_nibble as u8,

                    // LD Vx, K
                    // wait for a key press, store the value of the key in Vx
                    0x0A => self.wait_key_pressed(vx),

                    // LD DT, Vx
                    // Set delay timer = Vx
                    0x15 => self.delay_timer = self.registers[vx],

                    // LD ST, Vx
                    // Set sound timer = Vx
                    0x18 => self.sound_timer = self.registers[vx],

                    // ADD I, Vx
                    // Set I = I + Vx
                    0x1E => self.index_register += self.registers[vx] as u16,

                    // LD F, Vx
                    // Set I = location of the sprite for digit Vx
                    0x29 => self.set_index_to_sprite(vx),

                    // LD B, Vx
                    // Store BCD representation of Vx in memory locations I, I+1, and I+2
                    0x33 => self.store_bcd(vx),

                    // LD [I], Vx
                    // Store registers V0 through Vx in memory starting at I location
                    0x55 => self.store_registers(vx),

                    // LD Vx, [I]
                    // Read registers V0 through Vx from memory starting at location I
                    0x65 => self.load_registers(vx),

                    _ => unimplemented!("Opcode not implemented: {:04X}", opcode),
                }
            }

            //
            // _ => unimplemented!("Opcode not implemented: {:04X}", opcode),
            _ => panic!("Code not Known!! -> {:04X}", opcode),
        }
    }

    fn run_cycle(&mut self) {
        // Fetch, decode, and execute one instruction
        let opcode = self.fetch_opcode();
        self.execute_opcode(opcode);
        // Increment program counter
        self.program_counter += 2;


       
    }
    
}
 

struct Graphics {
    screen: [[bool; DISPLAY_WIDTH]; DISPLAY_HEIGHT],
}

impl Graphics {
    // Implementation of the Graphics struct methods...
}

fn render_graphics(graphics: &Graphics) {
    // Create a window
    let mut window = Window::new(
        "CHIP-8 Emulator",
        DISPLAY_WIDTH * WINDOW_SCALE,
        DISPLAY_HEIGHT * WINDOW_SCALE,
        WindowOptions::default(),
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });

    // Buffer for holding the pixels to be rendered
    let mut buffer = vec![0; DISPLAY_WIDTH * DISPLAY_HEIGHT * 4];

    while window.is_open() && !window.is_key_down(Key::Escape) {
        // Convert the graphics state to RGBA buffer for rendering
        for (i, row) in graphics.screen.iter().enumerate() {
            for (j, &pixel) in row.iter().enumerate() {
                let color = if pixel { 255 } else { 0 };
                let index = (i * DISPLAY_WIDTH + j) * 4;
                buffer[index] = color;
                buffer[index + 1] = color;
                buffer[index + 2] = color;
                buffer[index + 3] = 255; // Alpha channel
            }
        }

        // Render the buffer to the window
        if let Err(e) = window.update_with_buffer_size(&buffer, DISPLAY_WIDTH, DISPLAY_HEIGHT) {
            println!("Window update error: {}", e);
            break;
        }

        // Emulate a refresh rate (60 Hz)
        std::thread::sleep(Duration::from_micros(1000000 / 60));
    }
}





fn main() {
   
    // Get the ROM file path from command-line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <path/to/rom.ch8>", args[0]);
        std::process::exit(1);
    }
    let rom_file = &args[1];

    // Create and initialize the Chip8 emulator
    let mut chip8 = Chip8::new();

    // Load the ROM file
    if let Err(err) = chip8.load_rom(rom_file) {
        eprintln!("Error loading ROM file: {}", err);
        std::process::exit(1);
    }
    
    //chip8.clear_screen();

    let mut graphics = Graphics {
        screen: [[false; DISPLAY_WIDTH]; DISPLAY_HEIGHT],
    };

    // Create a window
    let mut window = Window::new(
        "CHIP-8 Emulator",
        DISPLAY_WIDTH * WINDOW_SCALE,
        DISPLAY_HEIGHT * WINDOW_SCALE,
        WindowOptions::default(),
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });


    // Buffer for holding the pixels to be rendered
    let mut buffer = vec![0; DISPLAY_WIDTH * WINDOW_SCALE * DISPLAY_HEIGHT * WINDOW_SCALE * 4];

    

    // Main emulation loop
    while window.is_open() && !window.is_key_down(Key::Escape) {
        // Emulate one cycle of the CHIP-8 CPU
        chip8.run_cycle();
        // Update the display based on the CPU state
        
        // Update display based on CPU state
        for y in 0..DISPLAY_HEIGHT {
            for x in 0..DISPLAY_HEIGHT {
                let pixel = graphics.screen[y][x];
                // Convert CHIP-8 pixel state to RGBA color value
                let color = if pixel { 255 } else { 0 };
                // Set the same color for all pixels within a scaled block
                for sy in 0..WINDOW_SCALE {
                    for sx in 0..WINDOW_SCALE {
                        let index = ((y * WINDOW_SCALE + sy) * DISPLAY_WIDTH * WINDOW_SCALE + (x * WINDOW_SCALE + sx)) * 4;
                        buffer[index] = color;
                        buffer[index + 1] = color;
                        buffer[index + 2] = color;
                        buffer[index + 3] = 255; // Alpha channel
                    }
                }
            }
        }

        // Render the buffer to the window
        if let Err(e) = window.update_with_buffer_size(&buffer, DISPLAY_WIDTH * WINDOW_SCALE, DISPLAY_HEIGHT * WINDOW_SCALE) {
            println!("Window update error: {}", e);
            break;
        }


        // Update input state (if needed)
        // Handle sound (if needed)


        /* 
        // Render the buffer to the window
        if let Err(e) = window.update_with_buffer(&buffer, SCREEN_WIDTH * WINDOW_SCALE, SCREEN_HEIGHT * WINDOW_SCALE) {
            println!("Window update error: {}", e);
            break;
        }
*/
        // Emulate a refresh rate (60 Hz)
        std::thread::sleep(Duration::from_micros(1000000 / 60));
    }

    /* 
    // Emulation loop
    loop {
        chip8.run_cycle();
        
        //TODO Update the Display based on CPU status
        render_graphics(&graphics);

        //TODO UPDATE input state

        // TODO Handle sound

        // TODO Render Graphics
    }*/
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clear_display() {
        let mut chip8 = Chip8::new();
        // Set some initial state (e.g., non-empty display buffer)
        // For simplicity, let's assume the display buffer is filled with true values
        for row in &mut chip8.display_buffer {
            for pixel in row.iter_mut() {
                *pixel = true;
            }
        }

        chip8.clear_screen(); // Clear Display

        // Assert that all pixels in the display buffer are false
        assert!(chip8
            .display_buffer
            .iter()
            .all(|row| row.iter().all(|&pixel| !pixel)));
    }

    #[test]
    fn test_jump() {
        let mut chip8 = Chip8::new();
        chip8.execute_opcode(0x1234); // Jump to address 0x234

        // Assert that the program counter was set to address 0x234
        assert_eq!(chip8.program_counter, 0x234);
    }

    #[test]
    fn test_jump_plus_v0() {
        let mut chip8 = Chip8::new();
        chip8.registers[0] = 0x10;
        chip8.execute_opcode(0xB234); // Jump to address 0x234 + V0

        // Assert that the program counter was set to address 0x234 + V0
        assert_eq!(chip8.program_counter, 0x244);
    }

    #[test]
    fn test_call_and_return() {
        let mut chip8 = Chip8::new();
        chip8.execute_opcode(0x2345); // Call address 0x345
        chip8.execute_opcode(0x00EE); // Return from subroutine

        // Assert that the program counter was restored after returning from subroutine
        assert_eq!(chip8.program_counter, 0x200);
    }

    #[test]
    fn test_skip_if_equal() {
        let mut chip8 = Chip8::new();
        chip8.registers[1] = 0x10;
        chip8.execute_opcode(0x3110); // Skip next instruction if V1 == 0x10

        // Assert that the program counter was incremented by 2
        assert_eq!(chip8.program_counter, 0x202);

        chip8.execute_opcode(0x3111); // Skip next instruction if V1 == 0x11

        // Assert that the program counter was not incremented
        assert_eq!(chip8.program_counter, 0x202);
    }

    // Add more tests for other display-related methods...
}
