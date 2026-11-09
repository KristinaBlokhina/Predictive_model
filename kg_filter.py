import numpy as np

# Усі члени полінома (ім'я, функція від x1, x2) у порядку Колмогорова–Габора
TERMS = [
    ("1",        lambda x1, x2: 1.0),
    ("x1",       lambda x1, x2: x1),
    ("x2",       lambda x1, x2: x2),
    ("x1*x2",    lambda x1, x2: x1 * x2),
    ("x1^2",     lambda x1, x2: x1 ** 2),
    ("x2^2",     lambda x1, x2: x2 ** 2),
    ("x1^2*x2",  lambda x1, x2: x1 ** 2 * x2),
    ("x1*x2^2",  lambda x1, x2: x1 * x2 ** 2),
    ("x1^2*x2^2", lambda x1, x2: x1 ** 2 * x2 ** 2),
]

# Набір опорних моделей: кількість перших членів поліному
MODELS = {
    "M1 (a0+a1x1)":                   [0, 1],          # лінійна, 1 лаг
    "M2 (a0+a1x1+a2x2)":              [0, 1, 2],       # лінійний фільтр, 2 лаги
    "M3 (a0+a1x1+a2x2+a3x1x2)":       [0, 1, 2, 3],    # опорна модель з методички
    "M4 (M3 + a4x1^2)":               [0, 1, 2, 3, 4],
    "M5 (M3 + a4x1^2 + a5x2^2)":      [0, 1, 2, 3, 4, 5],
}


def design(series, idx):
    """Матриця умовних рівнянь: рядок i = [члени(F[i], F[i+1])], права частина = F[i+2]."""
    s = np.asarray(series, float)
    rows = []
    for i in range(len(s) - 2):
        rows.append([TERMS[j][1](s[i], s[i + 1]) for j in idx])
    A = np.array(rows, float)
    y = s[2:].copy()
    return A, y


def fit(series, idx):
    """Нормалізація (Гаус): N = A^T A, b = A^T y; розв'язок N a = b."""
    A, y = design(series, idx)
    N, b = A.T @ A, A.T @ y
    if A.shape[0] >= A.shape[1] and np.linalg.matrix_rank(N) == N.shape[0]:
        a = np.linalg.solve(N, b)
        exact = True
    else:                       # недовизначена/виродженa система -> мінімальна норма
        a = np.linalg.lstsq(A, y, rcond=None)[0]
        exact = False
    return a, N, b, exact


def predict_one(a, idx, x1, x2):
    return float(sum(a[k] * TERMS[j][1](x1, x2) for k, j in enumerate(idx)))


def skv(series, a, idx):
    """СКВ у навчальних точках: sqrt( sum (Y_експ - Y_розр)^2 / n )."""
    A, y = design(series, idx)
    r = y - A @ a
    return float(np.sqrt(np.mean(r ** 2)))


def forecast_recursive(series_train, a, idx, steps):
    """Ітераційний прогноз на steps кроків: кожне нове значення додається в передісторію."""
    hist = list(map(float, series_train))
    out = []
    for _ in range(steps):
        v = predict_one(a, idx, hist[-2], hist[-1])
        out.append(v)
        hist.append(v)
    return out


def horizon(series, idx, tol_rel=0.15, kmax=None):"""Лабораторна робота №4. Прогнозуючий фільтр Колмогорова–Габора.

Опорна модель будується з членів полінома Колмогорова–Габора 2-го порядку
від двох аргументів x1 = F(n-1) (старіше значення), x2 = F(n) (новіше), як у числових рядках методички:
    Y = a0 + a1*x1 + a2*x2 + a3*x1*x2 + a4*x1^2 + a5*x2^2 + a6*x1^2*x2 + a7*x1*x2^2 + a8*x1^2*x2^2
Коефіцієнти знаходяться методом нормальних рівнянь Гауса (A^T A) a = A^T y.
"""
import numpy as np

# Усі члени полінома (ім'я, функція від x1, x2) у порядку Колмогорова–Габора
TERMS = [
    ("1",        lambda x1, x2: 1.0),
    ("x1",       lambda x1, x2: x1),
    ("x2",       lambda x1, x2: x2),
    ("x1*x2",    lambda x1, x2: x1 * x2),
    ("x1^2",     lambda x1, x2: x1 ** 2),
    ("x2^2",     lambda x1, x2: x2 ** 2),
    ("x1^2*x2",  lambda x1, x2: x1 ** 2 * x2),
    ("x1*x2^2",  lambda x1, x2: x1 * x2 ** 2),
    ("x1^2*x2^2", lambda x1, x2: x1 ** 2 * x2 ** 2),
]

# Набір опорних моделей: кількість перших членів поліному
MODELS = {
    "M1 (a0+a1x1)":                   [0, 1],          # лінійна, 1 лаг
    "M2 (a0+a1x1+a2x2)":              [0, 1, 2],       # лінійний фільтр, 2 лаги
    "M3 (a0+a1x1+a2x2+a3x1x2)":       [0, 1, 2, 3],    # опорна модель з методички
    "M4 (M3 + a4x1^2)":               [0, 1, 2, 3, 4],
    "M5 (M3 + a4x1^2 + a5x2^2)":      [0, 1, 2, 3, 4, 5],
}


def design(series, idx):
    """Матриця умовних рівнянь: рядок i = [члени(F[i], F[i+1])], права частина = F[i+2]."""
    s = np.asarray(series, float)
    rows = []
    for i in range(len(s) - 2):
        rows.append([TERMS[j][1](s[i], s[i + 1]) for j in idx])
    A = np.array(rows, float)
    y = s[2:].copy()
    return A, y


def fit(series, idx):
    """Нормалізація (Гаус): N = A^T A, b = A^T y; розв'язок N a = b."""
    A, y = design(series, idx)
    N, b = A.T @ A, A.T @ y
    if A.shape[0] >= A.shape[1] and np.linalg.matrix_rank(N) == N.shape[0]:
        a = np.linalg.solve(N, b)
        exact = True
    else:                       # недовизначена/виродженa система -> мінімальна норма
        a = np.linalg.lstsq(A, y, rcond=None)[0]
        exact = False
    return a, N, b, exact


def predict_one(a, idx, x1, x2):
    return float(sum(a[k] * TERMS[j][1](x1, x2) for k, j in enumerate(idx)))


def skv(series, a, idx):
    """СКВ у навчальних точках: sqrt( sum (Y_експ - Y_розр)^2 / n )."""
    A, y = design(series, idx)
    r = y - A @ a
    return float(np.sqrt(np.mean(r ** 2)))


def forecast_recursive(series_train, a, idx, steps):
    """Ітераційний прогноз на steps кроків: кожне нове значення додається в передісторію."""
    hist = list(map(float, series_train))
    out = []
    for _ in range(steps):
        v = predict_one(a, idx, hist[-2], hist[-1])
        out.append(v)
        hist.append(v)
    return out


def horizon(series, idx, tol_rel=0.15, kmax=None):
    """Навчання за n-k точками, прогноз k останніх точок (рекурсивно). Якщо похибка k-ї точки
    (відносна) <= tol_rel і всі попередні кроки теж прийнятні -- горизонт >= k.
    Повертає (горизонт, список рядків-протоколу)."""
    s = np.asarray(series, float)
    n = len(s)
    kmax = kmax or n
    log, H = [], 0
    for k in range(1, kmax + 1):
        m = n - k                       # довжина навчальної вибірки
        if m - 2 < len(idx):            # менше рівнянь, ніж невідомих
            log.append((k, m, None, None, "недостатньо даних"))
            break
        a, *_ = fit(s[:m], idx)
        pred = forecast_recursive(s[:m], a, idx, k)
        err = abs(pred[-1] - s[n - 1]) / abs(s[n - 1])
        ok = err <= tol_rel
        log.append((k, m, pred[-1], err, "прийнятна" if ok else "неприйнятна"))
        if ok:
            H = k
        else:
            break
    return H, log
    s = np.asarray(series, float)
    n = len(s)
    kmax = kmax or n
    log, H = [], 0
    for k in range(1, kmax + 1):
        m = n - k                       # довжина навчальної вибірки
        if m - 2 < len(idx):            # менше рівнянь, ніж невідомих
            log.append((k, m, None, None, "недостатньо даних"))
            break
        a, *_ = fit(s[:m], idx)
        pred = forecast_recursive(s[:m], a, idx, k)
        err = abs(pred[-1] - s[n - 1]) / abs(s[n - 1])
        ok = err <= tol_rel
        log.append((k, m, pred[-1], err, "прийнятна" if ok else "неприйнятна"))
        if ok:
            H = k
        else:
            break
    return H, log