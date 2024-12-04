import heapq

# 定義邏輯閘類別
class LogicGate:
    def __init__(self, name):
        self.name = name
        self.inputs = []
        self.output = None
        self.output_wire = None

    def evaluate(self):
        # 需要在子類別中實現具體的邏輯
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

    def schedule_event(self, event):
        heapq.heappush(self.event_queue, event)

    def run(self):
        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            self.time = event.time
            if event.wire.signal != event.signal:
                print(f"Time {self.time}: {event.wire.name} changes to {event.signal}")
                event.wire.signal = event.signal
                for gate in event.wire.targets:
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

# 添加邏輯閘
and_gate = sim.add_gate("AND", "G1")
and_gate.inputs = [a, b]
and_gate.output_wire = c

or_gate = sim.add_gate("OR", "G2")
or_gate.inputs = [c, out]
or_gate.output_wire = out

# 建立連線
a.connect(and_gate)
b.connect(and_gate)
c.connect(or_gate)
out.connect(or_gate)

# 安排初始事件
sim.schedule_event(Event(0, a, 1))
sim.schedule_event(Event(0, b, 0))
sim.schedule_event(Event(5, b, 1))

# 執行模擬
sim.run()
