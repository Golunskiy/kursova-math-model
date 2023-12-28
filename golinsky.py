import matplotlib.pyplot as plt
from matplotlib.patches import Circle


import pandas as pd
import numpy as np


pd.set_option('display.max_colwidth', None)
pd.options.display.max_colwidth = None
pd.options.display.max_columns = 100
pd.options.display.max_rows = 20
pd.options.display.width = 0

count_f = 0


def f(x):
    global count_f
    count_f += 1

    return (10 * (x[0] - x[1])**2 + (x[0]-1)**2)**4


def f_yo(x, R=0, yo=0):
    if isinstance(x, np.matrix):
        x = [x.item(0), x.item(1)]

    if yo == 0:
        return f(x)

    elif yo == 1:  # x^2+y^2 <= 4
        if x[0]**2 + x[1]**2 <= 4:
            return f(x)
        else:
            return f(x) + R * (4 - x[0]**2 - x[1]**2)**2

    elif yo == 2:  # x^2+y^2 <= 4  x^2+y^2 >= 1
        if x[0]**2 + x[1]**2 <= 4 and x[0]**2 + x[1]**2 >= 1:
            return f(x)
        else:
            return f(x) + R * (4 - x[0]**2 - x[1]**2)**2 + R * (x[0]**2 + x[1]**2 - 1)**2

    elif yo == 3:  # x^2+y^2 <= 0.81
        if x[0]**2 + x[1]**2 <= 0.81:
            return f(x)
        else:
            return f(x) + R * (0.81 - x[0]**2 - x[1]**2)**2


def grad_f(x, h, grad_schema, R=0, yo=0):
    n = len(x)
    grad = np.zeros(n)

    f_x = f_yo(x, R, yo)
    for i in range(n):
        x_minus = x.copy()
        x_plus = x.copy()
        x_minus[i] -= h
        x_plus[i] += h

        if grad_schema == -1:
            grad[i] = (f_x - f_yo(x_minus, R, yo)) / h
        elif grad_schema == 1:
            grad[i] = (f_yo(x_plus, R, yo) - f_x) / h
        else:
            grad[i] = (f_yo(x_plus, R, yo) - f_yo(x_minus, R, yo)) / (2 * h)

    return grad, f_x


def norma(x):
    if isinstance(x, np.matrix):
        x = [x.item(0), x.item(1)]
    return (x[0]**2 + x[1]**2)**(1/2)


def sven(x, s, l0, alpha=0.1):
    delta_l = alpha * (norma(x) / norma(s))
    l = [l0]
    fl = [f(x + l[0] * s)]

    l_i = 1
    l_1 = l[0] + delta_l
    l_2 = l[0] - delta_l
    fl_1 = f(x + l_1 * s)
    fl_2 = f(x + l_2 * s)

    if fl_1 > fl[0] and fl_2 > fl[0]:
        return [l_2, l_1]

    if fl_2 < fl_1:
        l_i = -1 * l_i
        l.append(l_2)
        fl.append(fl_2)
    else:
        l.append(l_1)
        fl.append(fl_1)

    while fl[-1] < fl[-2]:
        l_i *= 2
        l.append(l[-1] + l_i * delta_l)
        fl.append(f(x + l[-1] * s))

    l.append((l[-1]+l[-2])/2)
    fl.append(f(x + l[-1] * s))

    min_f_l = fl[0]
    for el_fl in fl[1:]:
        if el_fl < min_f_l:
            min_f_l = el_fl

    if fl[-1] == min_f_l:
        span = [l[-2], l[-3]]
    else:
        span = [l[-4], l[-1]]
    if span[0] > span[1]:
        span[0], span[1] = span[1], span[0]

    return span


def gold(x, s, e=0.01, alpha=0.1):
    l = [0]
    span = sven(x, s, 0, alpha)

    a = min(span)
    b = max(span)
    k = 0.618
    x_1 = a + (1 - k) * (b - a)
    x_2 = a + k * (b - a)

    f1, f2 = f(x + x_1 * s), f(x + x_2 * s)

    while abs(b - a) > e:
        if f1 < f2:
            b = x_2
            x_1 = a + (1 - k) * (b - a)
            x_2 = a + k * (b - a)
            f1, f2 = f(x + x_1 * s), f(x + x_2 * s)
        else:
            a = x_1
            x_1 = a + (1 - k) * (b - a)
            x_2 = a + k * (b - a)

            f1, f2 = f(x + x_1 * s), f(x + x_2 * s)

    # x.append(x[-1] + l[-1] * s)
    l.append((a + b) / 2)
    # return (a+b)/2

    # return x + l[-1] * s
    return l[-1]


def dsk_payell(x, s, e=0.01, alpha=0.1, max_count=5000):
    span = sven(x, s, 0, alpha)
    x_1 = min(span)
    x_3 = max(span)
    x_2 = (x_1 + x_3) / 2
    # print(x, s)
    # print('!!!!!!!!!!!!!!!!!!!!!')
    fx_1 = f(x + x_1 * s)
    fx_2 = f(x + x_2 * s)
    fx_3 = f(x + x_3 * s)
    while True:
        if count_f > max_count:
            return None

        # print(pn(x_1), pn(x_2), pn(x_3))
        # print(pn(fx_1), pn(fx_2), pn(fx_3))

        a1 = (fx_2 - fx_1) / (x_2 - x_1)
        a2 = 1 / (x_3 - x_2) * ((fx_3 - fx_1) / (x_3 - x_1) - (
                fx_2 - fx_1) / (x_2 - x_1))
        x_ = (x_1 + x_2) / 2 - a1 / (2 * a2)

        fx_ = f(x + x_ * s)

        # print(pn(x_), pn(fx_))
        condition_1 = abs(fx_2 - fx_) <= e
        condition_2 = abs(x_2 - x_) <= e
        condition = condition_1 and condition_2

        if not condition:
            xs = [x_1, x_2, x_3, x_]
            fs = [fx_1, fx_2, fx_3, fx_]

            # min_i = fs.index(min(fs))
            indexes = [i for i in range(len(fs)) if fs[i] == min(fs)]

            for ind in indexes:
                if xs[ind] != min(xs) and xs[ind] != max(xs):
                    min_i = ind
            if not min_i:
                print('!')
            xs_left = []
            xs_right = []

            for el in xs:
                if el > xs[min_i]:
                    xs_right.append(el)
                elif el < xs[min_i]:
                    xs_left.append(el)

            if len(xs_left) == 1:
                x_1 = xs_left[0]
            else:
                x_1 = max(xs_left)

            if len(xs_right) == 1:
                x_3 = xs_right[0]
            else:
                x_3 = min(xs_right)

            x_2 = xs[min_i]

            fx_1 = fs[xs.index(x_1)]
            fx_2 = fs[min_i]
            fx_3 = fs[xs.index(x_3)]

            continue

        # return x + x_ * s
        return x_


def cret1(x, f_x, e):
    one = ((x[-1][0] - x[-2][0]) ** 2 + (x[-1][1] - x[-2][1]) ** 2) ** (1 / 2) / (x[-2][0] ** 2 + x[-2][1] ** 2) ** (1 / 2)
    two = abs(f_x[-1] - f_x[-2])

    return one <= e and two <= e


def cret2(grad, e):
    cret = norma(grad)
    return cret <= e


def mop_method(mop, xk, s, mop_e, alpha, max_count):
    if mop == 'gold':
        alpha_k = gold(xk, s, mop_e, alpha)
    elif mop == 'dsk':
        alpha_k = dsk_payell(xk, s, mop_e, alpha, max_count)
    return alpha_k


def method_pirsona_3(x0, e=10**(-5), h=0.01, grad_schema=0, mop='gold', mop_e=0.1, alpha=0.1, cret=1, restart=False,
               k=10**(-7), R=0, yo=0, max_count=2000):
    global count_f
    count_f = 0
    # Ініціалізація матриці псевдо-гессіана та градієнта
    Ak = np.eye(len(x0))
    grad_fk, f_x0 = grad_f(x0, h, grad_schema, R, yo)
    xk = x0
    xk1 = None
    xs = [xk]
    fs = [f_x0]

    # Ітераційний процес
    while True:
        # Обчислення напрямку спуску
        s = -np.dot(Ak, grad_fk)
        if isinstance(s, np.matrix):
            s = np.array([s.item(0), s.item(1)])

        # Одновимірний пошук для знаходження довжини кроку alpha_k
        alpha_k = mop_method(mop, xk, s, mop_e, alpha, max_count)
        if alpha_k is None:
            break

        # Рестарт, якзо довжина кроку мала, або від'ємна
        if restart and alpha_k < k:
            Ak = np.eye(len(x0))
            s = -np.dot(Ak, grad_fk)

            # Одновимірний пошук для знаходження довжини кроку alpha_k
            alpha_k = mop_method(mop, xk, s, mop_e, alpha, max_count)
            if alpha_k is None:
                break

        # Обчислення нової точки та градієнта в цій точці
        xk1 = xk + alpha_k * s
        grad_fk1, f_xk = grad_f(xk1, h, grad_schema, R, yo)

        # Рестарт, якщо значення функції збільшується
        if restart and f_xk > fs[-1]:
            Ak = np.eye(len(x0))
            s = -np.dot(Ak, grad_fk)

            # Одновимірний пошук для знаходження довжини кроку alpha_k
            alpha_k = mop_method(mop, xk, s, mop_e, alpha, max_count)
            if alpha_k is None:
                break

            xk1 = xk + alpha_k * s
            grad_fk1, f_xk = grad_f(xk1, h, grad_schema, R, yo)

        xs.append(xk1)  # записуємо x
        fs.append(f_xk)  # записуємо використане значення функції

        # Перевірка на зупинку за критерієм
        if cret == 1 and cret1(xs, fs, e):
            break
        if cret == 2 and cret2(grad_fk1, e):
            break

        # Обчислення векторів delta_g та delta_x
        delta_x = np.matrix(xk1 - xk).transpose()
        delta_g = np.matrix(grad_fk1 - grad_fk).transpose()

        # Обчислення нової Ak
        first = Ak
        second = delta_x - Ak @ delta_g
        third = np.divide((Ak @ delta_g).transpose(), delta_g.transpose() @ Ak @ delta_g)
        fourth = second @ third

        # Оновлення змінних
        Ak = first + fourth
        xk = xk1
        grad_fk = grad_fk1

        if count_f > max_count:
            break

    res_count = count_f
    count_f = 0
    return {"x": xk1, "xs": xs, "count": res_count, 'h': h, 'schema': grad_schema, 'e': e, 'R': R, 'yo': yo,
            'cret': cret, 'mop': mop, 'mop_e': mop_e, 'alpha': alpha, 'restart': restart, 'k': k}


def level_lines(ax, x_min=-2.1, x_max=2.1, y_min=-2.1, y_max=2.1):
    # Обчислюємо розмір сітки для обчислення ліній рівня
    grid_size = 100
    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)

    # Обчислюємо значення функції на сітці
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = f([X, Y])

    # Малюємо лінії рівня
    ax.contour(X, Y, Z, levels=100, alpha=0.5)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


def bo_plot(res):
    x = [el[0] for el in res['xs']]
    y = [el[1] for el in res['xs']]
    fig, ax = plt.subplots()
    ax.plot(x, y, marker='o', linestyle='-')
    ax.plot(x[0], y[0], marker='o', color='black', markersize=8, label=f"({x[0]}, {y[0]})")
    ax.plot(x[-1], y[-1], marker='o', color='red', markersize=8, label=f"({round(x[-1], 6)}, {round(y[-1], 6)})")
    # level_lines(ax, -1.5, 1.5, -1, 1)
    plt.legend(fontsize='large')
    plt.grid()
    plt.show()


def func_h(x0, e, grad_schema, mop, mop_e, alpha, cret, restart):
    hs = [1/10**el for el in range(1, 10)]

    df = pd.DataFrame(columns=['h', 'count', 'x', 'f(x)'])
    for h in hs:
        obj = method_pirsona_3(x0, e, h, grad_schema, mop, mop_e, alpha, cret, restart)
        row = {'h': format(obj['h']), 'count': obj['count'], 'x': obj['x'], 'f(x)': f(obj['x'])}
        df.loc[len(df)] = row

    df['count'] = df['count'].astype(int)
    # df = df.sort_values('count')

    print(df)


def func_grad_schema(x0, e, h, mop, mop_e, alpha, cret, restart):
    schemas = [-1, 1, 0]

    df = pd.DataFrame(columns=['schema', 'count', 'x', 'f(x)'])
    for schema in schemas:
        obj = method_pirsona_3(x0, e, h, schema, mop, mop_e, alpha, cret, restart)
        if obj['schema'] == -1:
            _schema = 'ліва'
        elif obj['schema'] == 0:
            _schema = 'центральна'
        elif obj['schema'] == 1:
            _schema = 'права'
        row = {'schema': _schema, 'count': obj['count'], 'x': obj['x'], 'f(x)': f(obj['x'])}
        df.loc[len(df)] = row

    df['count'] = df['count'].astype(int)

    print(df)


def func_h_and_grad_schema(x0, e, mop, mop_e, alpha, cret, restart):
    hs = [1 / 10 ** el for el in range(1, 12)]
    big_hs = []
    for h in hs:
        for coef in [0, 0.25, 0.5, 0.75]:
            big_hs.append(h - h*coef)
    print(big_hs)

    left = []
    center = []
    right = []

    for h in big_hs:
        for schema in [-1, 0, 1]:
            obj = method_pirsona_3(x0, e, h, schema, mop, mop_e, alpha, cret, restart)

            if schema == -1:
                left.append(obj['count'])
            elif schema == 0:
                center.append(obj['count'])
            else:
                right.append(obj['count'])
    format_big_hs = ["{:.2e}".format(num) for num in big_hs]

    plt.plot(format_big_hs, left, label='left')
    plt.plot(format_big_hs, center, label='center')
    plt.plot(format_big_hs, right, label='right')
    plt.xticks(rotation=90)
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['legend.fontsize'] = 20
    plt.title('Залежність N від кроку і схеми похідних')
    plt.legend()
    plt.show()


def func_mop(x0, e, h, grad_schema, alpha, cret, restart):
    es = [1 / 10 ** item for item in range(1, 9)]

    if True:
        df_gold = pd.DataFrame()
        for mop_e in es:
            obj = method_pirsona_3(x0, e, h, grad_schema, 'gold', mop_e, alpha, cret, restart)
            new_row = {'e': format(obj['mop_e']), 'count': obj['count'], 'x': obj['x'], 'f(x)': f(obj['x'])}
            df_gold = df_gold.append(new_row, ignore_index=True)

        df_gold['count'] = df_gold['count'].astype(int)
        print('gold')
        print(df_gold)
        print()

    if True:
        df_dsk = pd.DataFrame()
        for mop_e in es:
            obj = method_pirsona_3(x0, e, h, grad_schema, 'dsk', mop_e, alpha, cret, restart)
            new_row = {'e': format(obj['mop_e']), 'count': obj['count'], 'x': obj['x'], 'f(x)': f(obj['x'])}
            df_dsk = df_dsk.append(new_row, ignore_index=True)

        df_dsk['count'] = df_dsk['count'].astype(int)
        print('dsk')
        print(df_dsk)
        print()


def func_sven(x0, e, h, grad_schema, mop, mop_e, cret, restart):
    alphases = [
        [3, 2, 1, *[1 / 10 ** item for item in range(1, 7)]],
        # [0.005, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14]
    ]
    for ind in [0]:
        alphas = alphases[ind]
        alphas.reverse()

        df_dsk = pd.DataFrame()
        for alpha in alphas:
            obj = method_pirsona_3(x0, e, h, grad_schema, mop, mop_e, alpha, cret, restart)
            new_row = {'a': format(obj['alpha']), 'count': obj['count'], 'x': obj['x'], 'f(x)': f(obj['x'])}
            df_dsk = df_dsk.append(new_row, ignore_index=True)

        df_dsk['count'] = df_dsk['count'].astype(int)
        print(df_dsk)
        N = df_dsk['count']
        a_e = df_dsk['a']
        t = df_dsk['f(x)']

        fig, ax = plt.subplots()
        for i in range(len(a_e)):
            if float(t[i]) > e:
                ax.bar(str(round(float(a_e[i]), 3)), N[i], color='red')
            else:
                ax.bar(str(round(float(a_e[i]), 3)), N[i], color='blue')
        ax.axhline(y=600, color='black')

        ax.set_xlabel('a')
        ax.set_ylabel('count')
        ax.xaxis.label.set_size(16)
        ax.yaxis.label.set_size(16)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_title('Залежність кількості обчислень f(x) від α', fontsize=18)
        plt.show()


def func_cret(x0, e, h, grad_schema, mop, mop_e, alpha, restart):
    df = pd.DataFrame(columns=['Критерій', 'count', 'x', 'f(x)'])

    for cret in [1, 2]:
        obj = method_pirsona_3(x0, e, h, grad_schema, mop, mop_e, alpha, cret, restart)

        new_row = {'Критерій': obj['cret'], 'count': obj['count'], 'x': obj['x'], 'f(x)': f(obj['x'])}
        df.loc[len(df)] = new_row

    df['count'] = df['count'].astype(int)
    df['Критерій'] = df['Критерій'].astype(int)
    print(df)


def func_restart(x0, e, h, grad_schema, mop, mop_e, alpha, cret):
    df = pd.DataFrame(columns=['restart', 'count', 'x', 'f(x)'])

    obj = method_pirsona_3(x0, e, h, grad_schema, mop, mop_e, alpha, cret, False)
    new_row = {'restart': 'Відсутні', 'count': obj['count'], 'x': obj['x'], 'f(x)': f(obj['x'])}
    df.loc[len(df)] = new_row

    obj = method_pirsona_3(x0, e, h, grad_schema, mop, mop_e, alpha, cret, True)
    new_row = {'restart': 'Наявні', 'count': obj['count'], 'x': obj['x'], 'f(x)': f(obj['x'])}
    df.loc[len(df)] = new_row

    df['count'] = df['count'].astype(int)
    print(df)


def func_e(x0, h, grad_schema, mop, mop_e, alpha, cret, restart, k=10**(-7)):
    es = [1/10**item for item in range(1, 15)]
    df = pd.DataFrame(columns=['e', 'count', 'x', 'f(x)'])

    for ei in es:
        obj = method_pirsona_3(x0, ei, h, grad_schema, mop, mop_e, alpha, cret, restart, k=k)
        new_row = {'e': format(obj['e']), 'count': obj['count'], 'x': obj['x'], 'f(x)': f(obj['x'])}
        df.loc[len(df)] = new_row

    df['count'] = df['count'].astype(int)
    # df['e'] = df['e'].astype(int)
    res_e = [str(el) for el in es]
    res_count = df['count'].tolist()

    # print(df)

    if False:
        fig, ax = plt.subplots()

        ax.bar(res_e, res_count)
        ax.axhline(y=600, color='r')

        ax.xaxis.label.set_size(16)
        ax.yaxis.label.set_size(16)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_title('Залежність кількості обчислень f(x) від точності', fontsize=18)
        plt.xlabel('Точність')
        plt.ylabel('Кількість обчислення f(x)')
        plt.show()

    return df


def func_k(x0, h, grad_schema, mop, mop_e, alpha, cret, restart):
    ks = [10, 5, 4, 3, 2, 1, *[1/10**num for num in range(1, 15)]]
    # ks = [num for num in range(50, 1000, 50)]
    # ks = [10 * num for num in range(100, 1001, 50)]

    df_main = pd.DataFrame()

    for k in ks:
        df = func_e(x0, h, grad_schema, mop, mop_e, alpha, cret, restart, k=k)
        df['e'] = df['e'].astype(float)
        # filtered_df = df[(df['e'] == df['e'].min()) & (df['count'] < 600)]

        df = df[(df['count'] < 600)]
        df = df[(df['e'] == df['e'].min())]

        df.insert(0, 'k', k)

        df_main = df_main.append(df, ignore_index=True)

    df_main['k'] = df_main['k'].apply(lambda x: '{:.0e}'.format(x) if x < 1 else int(x))
    df_main['e'] = df_main['e'].map('{:.0e}'.format)
    print(df_main)


def yo(x0, e, h, grad_schema, mop, mop_e, alpha, creterion, restart, k):
    R = 1
    df = pd.DataFrame(columns=['R', 'count', 'x', 'f(x)'])

    yo = 2

    x0 = [-2, -2]

    all_xs = [x0]

    fig, ax = plt.subplots()

    last_minimum = None

    while True:
        obj = method_pirsona_3(x0, e, h, grad_schema, mop, mop_e, alpha, creterion, restart, 10**(-5), R, yo)
        if obj['x'] is None:
            print('break')
            break
        new_row = {'R': obj['R'], 'count': obj['count'], 'x': obj['x'], 'f(x)': f(obj['x'])}
        df.loc[len(df)] = new_row

        x0 = obj['x']
        all_xs.append(obj['x'])

        x = [coord[0] for coord in obj['xs']]
        y = [coord[1] for coord in obj['xs']]
        ax.plot(x, y, marker='o', linestyle='-')

        if last_minimum is not None and abs(f(obj['x']) - f(last_minimum)) < e:
            break

        last_minimum = obj['x']
        R *= 10

    df['count'] = df['count'].astype(int)
    df['R'] = df['R'].astype(int)
    print(df)

    x = [coord[0] for coord in all_xs]
    y = [coord[1] for coord in all_xs]
    ax.plot(x, y, marker='o', linestyle='-', color='black')

    if yo == 1:
        circle = Circle((0, 0), 2, color='lightblue', fill=True, lw=0)
        ax.add_patch(circle)
        ax.plot(1, 1, color="red", marker='o')

    elif yo == 2:
        circle = Circle((0, 0), 2, color='lightblue', fill=True, lw=0)
        ax.add_patch(circle)
        circle = Circle((0, 0), 1, color='#ffffff', fill=True, lw=0)
        ax.add_patch(circle)
        ax.plot(1, 1, color="red", marker='o')

    elif yo == 3:
        circle = Circle((0, 0), 0.9, color='lightblue', fill=True, lw=0)
        ax.add_patch(circle)
        ax.plot(0.645086, 0.627586, color="red", marker='o')

    elif yo == 4:
        circle = Circle((0, 0), 0.9, color='lightblue', fill=True, lw=0)
        ax.add_patch(circle)
        circle = Circle((0, 0), 0.7, color='#ffffff', fill=True, lw=0)
        ax.add_patch(circle)
        ax.plot(0.728049, 0.529097, color="red", marker='o')

    ax.set_aspect("equal")
    plt.show()


x0 = np.array([-1.2, 0])
e = 1e-5  # точність
grad_schema = 0
mop = 'gold'
mop_e = 0.1
alpha = 0.1
cret = 1
restart = False

# ПОЧАТКОВИЙ ЗАПУСК
# res = method_pirsona_3(x0)
# print(f'x: {res["x"]}')
# print(f"f(x): {f(res['x'])}")
# print(f"count: {res['count']}")
# bo_plot(res)

# Крок похідних
func_h(x0, e, grad_schema, mop, mop_e, alpha, cret, restart)
h = 0.001

# Схема похідних
# func_grad_schema(x0, e, h, mop, mop_e, alpha, cret, restart)
grad_schema = 0

# МОП
# func_mop(x0, e, h, grad_schema, alpha, cret, restart)
mop = 'dsk'
mop_e = 0.00001

# Свен
# func_sven(x0, e, h, grad_schema, mop, mop_e, cret, restart)
alpha = 0.1

# Критерій закінчення
# func_cret(x0, e, h, grad_schema, mop, mop_e, alpha, restart)
cret = 2

# Рестарти та точність
# func_restart(x0, e, h, grad_schema, mop, mop_e, alpha, cret)
# func_e(x0, h, grad_schema, mop, mop_e, alpha, cret, False)
# func_e(x0, h, grad_schema, mop, mop_e, alpha, cret, True)
restart = True

# k
# func_k(x0, h, grad_schema, mop, mop_e, alpha, cret, restart)
k = 1
e = 10**(-12)

# УМОВНА ОПТИМІЗАЦІЯ
# yo(x0, e, h, grad_schema, mop, mop_e, alpha, cret, restart, k)

e = 10**(-14)
x0 = np.array([-1.2, 0])
# КІНЦЕВИЙ ЗАПУСК
# res = method_pirsona_3(x0, e, h, grad_schema, mop, mop_e, alpha, cret, restart, k)
# print(f'x: {res["x"]}')
# print(f"f(x): {f(res['x'])}")
# print(f"count: {res['count']}")
# bo_plot(res)

