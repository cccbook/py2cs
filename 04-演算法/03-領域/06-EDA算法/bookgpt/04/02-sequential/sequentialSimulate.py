import heapq

# 定義邏輯閘類別
class LogicGate:
    def __init__(self, name):
        self.name = name
        self.inputs = []
        self.output = None
        self.output_wire = None

    def evaluate(self):
        raise NotImplementedError

class AndGate(LogicGate):
    def evaluate(self):
        return all(wire.signal for wire in self.inputs)

class OrGate(LogicGate):
    def evaluate(self):
        return any(wire.signal for wire in self.inputs)

# 定義連線 (Wire)
class Wire:
    def __init__(self, name):
        self.name = name
        self.signal = 0
        self.targets = []

    def connect(self, target_gate):
        self.targets.append(target_gate)

# 定義暫存器 (Register)
class Register:
    def __init__(self, name):
        self.name = name
        self.input_wire = None
        self.output_wire = None
        self.state = 0  # 暫存器內部狀態

    def update(self):
        # 更新暫存器狀態
        self.state = self.input_wire.signal
        return self.state

# 定義事件
class Event:
    def __init__(self, time, wire, signal):
        self.time = time
        self.wire = wire
        self.signal = signal

    def __lt__(self, other):
        return self.time < other.time

# 定義事件驅動模擬器
class Simulator:
    def __init__(self):
        self.time = 0
        self.event_queue = []
        self.wires = {}
        self.gates = {}
        self.registers = []
        self.clock_period = 2  # 時脈週期（假設固定）

    def add_wire(self, name):
        wire = Wire(name)
        self.wires[name] = wire
        return wire

    def add_gate(self, gate_type, name):
        if gate_type == "AND":
            gate = AndGate(name)
        elif gate_type == "OR":
            gate = OrGate(name)
        else:
            raise ValueError(f"Unknown gate type: {gate_type}")
        self.gates[name] = gate
        return gate

    def add_register(self, name):
        reg = Register(name)
        self.registers.append(reg)
        return reg

    def schedule_event(self, event):
        heapq.heappush(self.event_queue, event)

    def schedule_clock(self):
        # 安排初始時脈事件
        self.schedule_event(Event(0, None, "CLOCK"))

    def run(self, time_max):
        self.schedule_clock()
        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            self.time = event.time
            if self.time > time_max: return

            if event.wire is None and event.signal == "CLOCK":
                # 處理時脈事件
                print(f"Time {self.time}: Clock edge")
                for reg in self.registers:
                    new_signal = reg.update()
                    if reg.output_wire.signal != new_signal:
                        self.schedule_event(Event(self.time + 1, reg.output_wire, new_signal))
                # 安排下一個時脈事件
                self.schedule_event(Event(self.time + self.clock_period, None, "CLOCK"))
            else:
                # 處理一般事件
                if event.wire.signal != event.signal:
                    print(f"Time {self.time}: {event.wire.name} changes to {event.signal}")
                    event.wire.signal = event.signal
                    for gate in event.wire.targets:
                        if isinstance(gate, LogicGate):  # 確保是邏輯閘
                            new_signal = gate.evaluate()
                            if gate.output_wire.signal != new_signal:
                                self.schedule_event(Event(self.time + 1, gate.output_wire, new_signal))

# 建立模擬器實例
sim = Simulator()

# 添加連線
a = sim.add_wire("A")
b = sim.add_wire("B")
c = sim.add_wire("C")
out = sim.add_wire("OUT")
reg_out = sim.add_wire("REG_OUT")

# 添加邏輯閘
and_gate = sim.add_gate("AND", "G1")
and_gate.inputs = [a, b]
and_gate.output_wire = c

# 添加暫存器
reg = sim.add_register("R1")
reg.input_wire = c
reg.output_wire = reg_out

# 添加 OR 閘連接暫存器輸出
or_gate = sim.add_gate("OR", "G2")
or_gate.inputs = [reg_out, a]
or_gate.output_wire = out

# 建立連線
a.connect(and_gate)
b.connect(and_gate)
c.connect(reg)
reg_out.connect(or_gate)
a.connect(or_gate)

# 安排初始事件
sim.schedule_event(Event(0, a, 1))
sim.schedule_event(Event(0, b, 0))
sim.schedule_event(Event(5, b, 1))

# 執行模擬
sim.run(50)
