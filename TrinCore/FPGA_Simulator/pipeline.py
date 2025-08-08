from typing import List, Dict, Any
from .verilog_generator import generate_verilog_module
import time

class PipelineSimulator:
    """Simulates a pipelined ternary CPU architecture"""
    
    def __init__(self, stages: int = 5):
        self.stages = stages
        self.pipeline = [None] * stages
        self.clock_cycle = 0
        self.throughput = 0
        self.latency = 0
        self.stall_count = 0
        
    def add_instruction(self, instruction: Dict[str, Any]):
        """Add an instruction to the pipeline"""
        if self.pipeline[0] is not None:
            self.stall_count += 1
            return False
        self.pipeline[0] = instruction
        return True
    
    def cycle(self):
        """Advance the pipeline by one cycle"""
        self.clock_cycle += 1
        
        # Move instructions through pipeline
        for i in range(self.stages-1, 0, -1):
            self.pipeline[i] = self.pipeline[i-1]
        self.pipeline[0] = None
        
        # Update metrics if instruction completed
        if self.pipeline[-1] is not None:
            self.throughput += 1
            self.latency = self.clock_cycle / max(1, self.throughput)
        
        return self.pipeline[-1]
    
    def simulate(self, program: List[Dict[str, Any]]):
        """Simulate a program through the pipeline"""
        results = []
        pc = 0
        
        while pc < len(program) or any(stage is not None for stage in self.pipeline):
            if pc < len(program):
                self.add_instruction(program[pc])
                pc += 1
            completed = self.cycle()
            if completed:
                results.append(completed)
        
        return {
            'results': results,
            'clock_cycles': self.clock_cycle,
            'throughput': self.throughput,
            'avg_latency': self.latency,
            'stall_count': self.stall_count,
            'cpi': self.clock_cycle / len(program)
        }

    def generate_pipelined_verilog(self):
        """Generate Verilog with pipeline support"""
        verilog = generate_verilog_module()
        verilog += """
// Pipeline registers
reg [1:0] if_id_instruction;
reg [1:0] id_ex_instruction;
reg [1:0] ex_mem_instruction;
reg [1:0] mem_wb_instruction;

always @(posedge clk) begin
    // IF stage
    if_id_instruction <= memory[pc];
    
    // ID stage
    id_ex_instruction <= if_id_instruction;
    
    // EX stage
    ex_mem_instruction <= id_ex_instruction;
    
    // MEM stage
    mem_wb_instruction <= ex_mem_instruction;
    
    // WB stage
    registers[mem_wb_instruction[4:2]] <= mem_wb_result;
end
"""
        return verilog
