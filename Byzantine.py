import hashlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
import matplotlib.animation as animation
import csv
import os

# ===================== CORE PARAMETERS =====================
Q_SHIFT = 16
N_NODES = 100_000
THRESHOLD_TAU = 0.7
ALPHA_BANACH = 0.05
TRAIL_LENGTH = 10
CSV_LOG_FILE = "network_audit_log.csv"

# ===================== FIXED POINT HELPERS =====================
def to_fixed(val):
    return np.array(val * (1 << Q_SHIFT), dtype=np.int32)

def from_fixed(val):
    return val.astype(np.float32) / (1 << Q_SHIFT)

def fixed_mul(a, b):
    return ((a.astype(np.int64) * b.astype(np.int64)) >> Q_SHIFT).astype(np.int32)

# ===================== NETWORK CLASS =====================
class DIM_Sovereign_Interactive:
    def __init__(self, byz_ratio=0.3, attack_type='drift'):
        np.random.seed(888)
        self.BYZANTINE_RATIO = byz_ratio
        self.attack_type = attack_type
        self.states = to_fixed(np.random.normal(0, 5, (N_NODES, 2)))
        self.prev_hash = b"GENESIS"
        self.novelty = 1.0
        self.audit_score = 1.0
        self.trail_buffer = np.zeros((TRAIL_LENGTH, N_NODES, 2), dtype=np.float32)
        self.trail_index = 0
        self.hist_met, self.hist_aud = [], []

        # CSV Logging
        if not os.path.exists(CSV_LOG_FILE):
            with open(CSV_LOG_FILE, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Byz_Ratio", "Attack_Type", "Audit_Score"])

        # Figure setup
        self.fig = plt.figure(figsize=(18, 10), facecolor='#020202')
        gs = self.fig.add_gridspec(2, 3, wspace=0.3, hspace=0.3)
        self._setup_axes(gs)
        self._setup_widgets()

    # ===================== VISUAL SETUP =====================
    def _setup_axes(self, gs):
        self.ax_sim = self.fig.add_subplot(gs[0, :2], facecolor='black')
        self.scatter = self.ax_sim.scatter([], [], s=0.4, animated=True)
        self.ax_sim.set_xlim(-25, 25)
        self.ax_sim.set_ylim(-25, 25)
        self.ax_sim.axis('off')
        self.ax_sim.set_title("DIM SOVEREIGN MANIFOLD", color='#00FFAD')

        self.ax_log = self.fig.add_subplot(gs[:, 2], facecolor='#0a0a0a')
        self.log_text = self.ax_log.text(0.05, 0.95, "", color='#00FFAD', family='monospace',
                                         fontsize=9, va='top')
        self.ax_log.axis('off')
        self.ax_log.set_title("DIM LEDGER", color='#00FFAD')

        self.ax_met = self.fig.add_subplot(gs[1, 0], facecolor='#050505')
        self.ax_aud = self.fig.add_subplot(gs[1, 1], facecolor='#050505')
        self.line_met, = self.ax_met.plot([], [], color='#00FFAD', lw=1.5)
        self.line_aud, = self.ax_aud.plot([], [], color='#FF3131', lw=1.5)
        self.ax_met.set_xlim(0, 50)
        self.ax_met.set_ylim(0, 2)
        self.ax_met.set_title("METABOLIC ACTIVITY", color='#00FFAD')
        self.ax_aud.set_xlim(0, 50)
        self.ax_aud.set_ylim(0, 1.1)
        self.ax_aud.set_title("MANIFOLD INTEGRITY", color='#FF3131')

    def _setup_widgets(self):
        # Byzantine ratio slider
        self.slider_ax = self.fig.add_axes([0.15, 0.02, 0.3, 0.03], facecolor='#222222')
        self.slider_byz = Slider(self.slider_ax, 'Byzantine Ratio', 0.0, 0.6,
                                 valinit=self.BYZANTINE_RATIO, valstep=0.01)
        self.slider_byz.on_changed(self.update_byz)

        # Attack type radio buttons
        self.radio_ax = self.fig.add_axes([0.55, 0.01, 0.15, 0.07], facecolor='#222222')
        self.radio = RadioButtons(self.radio_ax, ('drift', 'opposing', 'split', 'oscillate'))
        self.radio.set_active(['drift', 'opposing', 'split', 'oscillate'].index(self.attack_type))
        self.radio.on_clicked(self.update_attack)

    # ===================== CORE LOGIC =====================
    def _weiszfeld_median(self, points):
        m = np.mean(points, axis=0)
        for _ in range(3):
            dists = np.linalg.norm(points - m, axis=1)
            dists[dists == 0] = 1e-9
            w = 1.0 / dists
            m = np.sum(points * w[:, np.newaxis], axis=0) / np.sum(w)
        return m.astype(np.int32)

    def _apply_byzantine(self, step_idx):
        n_byz = int(N_NODES * self.BYZANTINE_RATIO)
        honest_states = self.states[n_byz:]
        if self.attack_type == 'drift':
            self.states[:n_byz] += to_fixed(np.random.normal(-0.5, 0.1, (n_byz, 2)))
        elif self.attack_type == 'opposing':
            centroid = np.mean(from_fixed(honest_states), axis=0)
            self.states[:n_byz] = to_fixed(-from_fixed(self.states[:n_byz] - centroid))
        elif self.attack_type == 'split':
            half = n_byz // 2
            self.states[:half] += to_fixed(np.array([1.0, 1.0]))
            self.states[half:n_byz] += to_fixed(np.array([-1.0, -1.0]))
        elif self.attack_type == 'oscillate':
            direction = 1 if step_idx % 2 == 0 else -1
            self.states[:n_byz] += to_fixed(np.random.normal(-0.5 * direction, 0.1, (n_byz, 2)))
        return n_byz

    # ===================== ANIMATION UPDATE =====================
    def update(self, frame):
        self.novelty = (self.novelty * 0.92) + (1.5 if frame % 30 == 0 else 0.05)
        active = self.novelty > THRESHOLD_TAU

        n_byz = self._apply_byzantine(frame)
        centroid = self._weiszfeld_median(self.states)
        self.states[n_byz:] += fixed_mul(to_fixed(ALPHA_BANACH), centroid - self.states[n_byz:])

        dists = np.linalg.norm(from_fixed(self.states - centroid), axis=1)
        self.audit_score = np.mean(dists < 3.0)

        sig = self.states[n_byz:, 0].tobytes()
        curr_hash = hashlib.sha256(sig + self.prev_hash).hexdigest()
        self.prev_hash = curr_hash.encode()

        # Update trail
        self.trail_buffer[self.trail_index % TRAIL_LENGTH] = from_fixed(self.states)
        self.trail_index += 1
        trails = np.mean(self.trail_buffer, axis=0)

        color_map = np.array(['#FF3131'] * n_byz + ['#00FFAD'] * (N_NODES - n_byz))
        self.scatter.set_offsets(trails)
        self.scatter.set_color(color_map)

        # Update history
        self.hist_met.append(self.novelty)
        self.hist_aud.append(self.audit_score)
        if len(self.hist_met) > 50: self.hist_met.pop(0)
        if len(self.hist_aud) > 50: self.hist_aud.pop(0)
        self.line_met.set_data(range(len(self.hist_met)), self.hist_met)
        self.line_aud.set_data(range(len(self.hist_aud)), self.hist_aud)

        # ===================== FILL AREAS SAFE =====================
        for coll in self.ax_met.collections[:]:
            coll.remove()
        self.ax_met.fill_between(range(len(self.hist_met)), 0, self.hist_met, color='#00FFAD', alpha=0.1)

        for coll in self.ax_aud.collections[:]:
            coll.remove()
        self.ax_aud.fill_between(range(len(self.hist_aud)), 0, self.hist_aud, color='#FF3131', alpha=0.1)

        # Collapse background highlight
        self.ax_sim.set_facecolor('#330000' if self.audit_score < 0.5 else 'black')

        status = "COLLAPSED!" if self.audit_score < 0.5 else ("INFLATED" if active else "DORMANT")
        log_text = (f"STATUS: {status}\nINTEGRITY: {self.audit_score*100:.2f}%\n"
                    f"BYZANTINE NODES: {n_byz}\nATTACK TYPE: {self.attack_type}\n\n"
                    f"LATEST HASH:\n{curr_hash[:32]}...\nCENTROID:\n{centroid}")
        self.log_text.set_text(log_text)

        # Log CSV
        with open(CSV_LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([self.BYZANTINE_RATIO, self.attack_type, self.audit_score])

        return self.scatter, self.line_met, self.line_aud, self.log_text

    # ===================== WIDGET CALLBACKS =====================
    def update_byz(self, val):
        self.BYZANTINE_RATIO = val

    def update_attack(self, label):
        self.attack_type = label

    # ===================== RUN ANIMATION =====================
    def run(self):
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=30,
                                           blit=True, cache_frame_data=False)
        plt.show()


# ===================== EXECUTE =====================
if __name__ == "__main__":
    substrate = DIM_Sovereign_Interactive(byz_ratio=0.3, attack_type='drift')
    substrate.run()
