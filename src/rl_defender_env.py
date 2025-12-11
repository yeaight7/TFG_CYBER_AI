import numpy as np
import gymnasium as gym
from gymnasium import spaces

class RLDatasetDefenderEnv(gym.Env):
    """
    Entorno RL para un defensor que decide PERMIT/BLOCK sobre muestras etiquetadas
    Cada step:
      - Observación: vector de características de una muestra (X[i])
      - Acción: 0 = PERMIT, 1 = BLOCK
      - Recompensa:
          * Ataque (label == attack_label):
                BLOCK  -> recompensa
                PERMIT -> penalización fuerte
          * Benigno (label == benign_label):
                PERMIT -> recompensa muy pequeña
                BLOCK  -> penalización

    Episodio: recorre hasta max_steps_per_episode muestras (o hasta agotar el dataset).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        benign_label: int = 0,
        attack_label: int = 1,
        correct_reward: float = 1.0,
        false_positive_penalty: float = -1.0,
        false_negative_penalty: float = -2.0,
        max_steps_per_episode: int | None = None,
        shuffle: bool = True,
    ) -> None:
        super().__init__()

        # Datos (n_samples, n_features)
        if X.ndim != 2:
            raise ValueError(f"X debe tener shape (n_samples, n_features), recibido {X.shape}")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("y debe ser 1D y tener el mismo número de muestras que X")

        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.n_samples, self.n_features = self.X.shape

        self.benign_label = int(benign_label)
        self.attack_label = int(attack_label)

        self.correct_reward = float(correct_reward)
        self.false_positive_penalty = float(false_positive_penalty)
        self.false_negative_penalty = float(false_negative_penalty)

        self.shuffle = bool(shuffle)
        self.max_steps_per_episode = max_steps_per_episode or self.n_samples

        # Espacios de Gymnasium
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_features,),
            dtype=np.float32,
        )

        # 0 = PERMIT, 1 = BLOCK
        self.action_space = spaces.Discrete(2)

        # Estado interno
        self.current_idx: int = 0
        self.steps: int = 0
        self.indices = np.arange(self.n_samples, dtype=np.int64)

    # Gymnasium API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        if self.shuffle:
            self.np_random.shuffle(self.indices)

        self.current_idx = 0
        self.steps = 0

        obs = self._get_observation()
        info: dict = {}
        return obs, info

    def _get_observation(self) -> np.ndarray:
        idx = self.indices[self.current_idx]
        return self.X[idx]

    def step(self, action: int):
        # Validación de acción
        if not self.action_space.contains(action):
            raise ValueError(f"Acción inválida: {action}")

        idx = self.indices[self.current_idx]
        label = self.y[idx]

        reward = 0.0

        is_attack = (label == self.attack_label)
        is_benign = (label == self.benign_label)

        # 0 = PERMIT, 1 = BLOCK
        if is_attack:
            if action == 1:  # bloquear ataque
                reward = self.correct_reward
            else:            # permitir ataque (FN)
                reward = self.false_negative_penalty
        elif is_benign:
            if action == 0:  # permitir benigno
                reward = self.correct_reward * 0.5
            else:            # bloquear benigno (FP)
                reward = self.false_positive_penalty
        else:
            # **expandir en caso de que haya más clases
            # Por defecto, las clases desconocidas serán consideradas ATAQUES (más vale prevenir que curar).
            if action == 1:  # bloquear (suspicious)
                reward = self.correct_reward
            else:            # permitir (suspicious) --- risky
                reward = self.false_negative_penalty

        # Avanzar al siguiente índice
        self.current_idx += 1
        self.steps += 1

        terminated = (self.current_idx >= self.n_samples)
        truncated = (self.steps >= self.max_steps_per_episode)

        if not (terminated or truncated):
            obs = self._get_observation()
        else:
            # Gymnasium exige devolver obs válida incluso al terminar
            obs = self.X[idx]

        info = {
            "sample_index": int(idx),
            "true_label": int(label),
        }

        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        # Opcional para debug
        pass

    def close(self):
        pass
