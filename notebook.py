import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import io

    return io, mo, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    upload = mo.ui.file(
        filetypes=[".csv"],
        label="Загрузи exoplanets_mass_period.csv"
    )

    upload
    return (upload,)


@app.cell(hide_code=True)
def _(io, mo, pd, upload):
    if upload.value:
        uploaded_file = upload.value[0]
        df = pd.read_csv(io.BytesIO(uploaded_file.contents))

        print("Данные успешно загружены!")
        print("Количество наблюдений:", len(df))
        print("Названия столбцов:", list(df.columns))

        df.head()
    else:
        mo.md("Сначала загрузи CSV-файл.")
    return (df,)


@app.cell(hide_code=True)
def _(df, np, pd):
    periods = pd.to_numeric(
        df["Период обращения (сутки)"],
        errors="coerce"
    ).dropna()

    periods = periods[periods > 0].to_numpy()

    if len(periods) == 0:
        raise ValueError("В данных не найдено положительных значений периода.")

    sorted_periods = np.sort(periods)

    print("Количество планет:", len(sorted_periods))
    print("Минимальный период:", round(sorted_periods.min(), 3), "суток")
    print("Максимальный период:", round(sorted_periods.max(), 3), "суток")
    print("Средний период:", round(sorted_periods.mean(), 3), "суток")
    return periods, sorted_periods


@app.cell(hide_code=True)
def _(mo):
    n_slider = mo.ui.slider(
        start=1000,
        stop=100000,
        step=1000,
        value=100000,
        label="Количество генерируемых значений"
    )

    seed_slider = mo.ui.slider(
        start=1,
        stop=1000,
        step=1,
        value=42,
        label="Начальное значение генератора (Seed)"
    )

    mo.vstack([
        n_slider,
        seed_slider
    ])
    return n_slider, seed_slider


@app.cell(hide_code=True)
def _(n_slider, np, pd, seed_slider, sorted_periods):
    rng = np.random.default_rng(seed_slider.value)

    uniform_values = rng.random(n_slider.value)

    indices = np.floor(
        uniform_values * len(sorted_periods)
    ).astype(int)

    generated_periods = sorted_periods[indices]

    print("Генерация завершена!")
    print("Количество значений:", len(generated_periods))

    pd.DataFrame({
        "Сгенерированный период (сутки)": generated_periods[:10]
    })
    return (generated_periods,)


@app.cell(hide_code=True)
def _(generated_periods, np, pd, periods):
    metrics = {
        "Среднее": [
            np.mean(periods),
            np.mean(generated_periods)
        ],
        "Медиана": [
            np.median(periods),
            np.median(generated_periods)
        ],
        "Стандартное отклонение": [
            np.std(periods),
            np.std(generated_periods)
        ],
        "Дисперсия": [
            np.var(periods),
            np.var(generated_periods)
        ],
        "25% квантиль": [
            np.quantile(periods, 0.25),
            np.quantile(generated_periods, 0.25)
        ],
        "75% квантиль": [
            np.quantile(periods, 0.75),
            np.quantile(generated_periods, 0.75)
        ]
    }

    metrics_df = pd.DataFrame(
        metrics,
        index=["Реальные данные", "Генератор"]
    ).T

    metrics_df
    return


@app.cell(hide_code=True)
def _(generated_periods, np, periods, plt):
    def plot_distribution_comparison():
        figure, axis = plt.subplots(figsize=(11, 5))

        bins = np.logspace(
            np.log10(periods.min()),
            np.log10(periods.max()),
            50
        )

        axis.hist(
            periods,
            bins=bins,
            weights=np.ones(len(periods)) / len(periods),
            alpha=0.5,
            label="Реальные данные",
            color="#91AFFF"
        )

        axis.hist(
            generated_periods,
            bins=bins,
            weights=np.ones(len(generated_periods)) / len(generated_periods),
            alpha=0.5,
            label="Сгенерированные данные",
            color="#72E5C2"
        )

        axis.set_xscale("log")
        axis.set_title("Сравнение реального и сгенерированного распределений")
        axis.set_xlabel("Период обращения (сутки, логарифмическая шкала)")
        axis.set_ylabel("Доля наблюдений")
        axis.legend()
        axis.grid(alpha=0.25)

        plt.show()

    plot_distribution_comparison()
    return


@app.cell(hide_code=True)
def _(generated_periods, np, periods, plt):
    def plot_log_distribution():
        figure, axis = plt.subplots(figsize=(11, 5))

        axis.hist(
            np.log10(periods),
            bins=50,
            density=True,
            alpha=0.5,
            label="Реальные данные",
            color="#91AFFF"
        )

        axis.hist(
            np.log10(generated_periods),
            bins=50,
            density=True,
            alpha=0.5,
            label="Сгенерированные данные",
            color="#72E5C2"
        )

        axis.set_title("Распределения в логарифмической шкале")
        axis.set_xlabel("log10(периода обращения)")
        axis.set_ylabel("Плотность")
        axis.legend()
        axis.grid(alpha=0.25)

        plt.show()

    plot_log_distribution()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Вывод

    В работе реализован генератор случайных значений
    на основе эмпирического распределения периодов
    обращения экзопланет.

    Для генерации использован метод обратной эмпирической
    функции распределения. Он позволяет получать новые
    значения, воспроизводящие распределение исходных данных.

    Сравнение статистических характеристик и гистограмм
    позволяет оценить сходство реальных и сгенерированных
    данных.

    При увеличении количества генерируемых значений
    статистические характеристики генератора должны
    приближаться к характеристикам исходной выборки.
    """)
    return


@app.cell(hide_code=True)
def _(generated_periods, np, pd, periods):
    # Сравнение статистических характеристик генератора

    real_metrics = {
        "Среднее": np.mean(periods),
        "Медиана": np.median(periods),
        "Дисперсия": np.var(periods),
        "Стандартное отклонение": np.std(periods),
        "Минимум": np.min(periods),
        "Максимум": np.max(periods),
        "25% квантиль": np.quantile(periods, 0.25),
        "75% квантиль": np.quantile(periods, 0.75),
        "90% квантиль": np.quantile(periods, 0.90)
    }

    generated_metrics = {
        "Среднее": np.mean(generated_periods),
        "Медиана": np.median(generated_periods),
        "Дисперсия": np.var(generated_periods),
        "Стандартное отклонение": np.std(generated_periods),
        "Минимум": np.min(generated_periods),
        "Максимум": np.max(generated_periods),
        "25% квантиль": np.quantile(generated_periods, 0.25),
        "75% квантиль": np.quantile(generated_periods, 0.75),
        "90% квантиль": np.quantile(generated_periods, 0.90)
    }

    comparison_df = pd.DataFrame({
        "Реальные данные": real_metrics,
        "Генератор": generated_metrics
    })

    comparison_df["Абсолютное отклонение"] = abs(
        comparison_df["Реальные данные"] -
        comparison_df["Генератор"]
    )

    comparison_df["Относительное отклонение, %"] = (
        comparison_df["Абсолютное отклонение"] /
        comparison_df["Реальные данные"].abs()
    ) * 100

    comparison_df.round(4)
    return (comparison_df,)


@app.cell(hide_code=True)
def _(comparison_df, np):
    mean_relative_error = comparison_df[
        "Относительное отклонение, %"
    ].replace([np.inf, -np.inf], np.nan).dropna().mean()

    print("Среднее относительное отклонение:", round(mean_relative_error, 2), "%")

    if mean_relative_error < 5:
        print("Результат: статистические характеристики генератора очень близки к реальным.")
    elif mean_relative_error < 15:
        print("Результат: статистические характеристики генератора достаточно близки к реальным.")
    else:
        print("Результат: имеются заметные различия между реальными и сгенерированными данными.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Вывод по качеству генератора

    В результате сравнения реальных и сгенерированных данных установлено, что статистические характеристики генератора достаточно близки к характеристикам исходной выборки.

    Относительное отклонение среднего составило 2,55%, медианы — 1,53%, дисперсии — 3,38%. Наименьшие отклонения наблюдаются у квантилей: для 25% квантиля — 0,08%, для 75% — 0,11%, для 90% — 0,69%.

    Минимальное и максимальное значения также совпали с исходными данными.

    Таким образом, при генерации 100 000 значений удалось получить выборку, которая хорошо воспроизводит основные статистические характеристики эмпирического распределения периодов обращения экзопланет.
    """)
    return


@app.cell(hide_code=True)
def _(generated_periods, n_slider, np, periods, seed_slider, sorted_periods):
    def check_generator():
        # Проверка 1. Количество значений
        test_size = len(generated_periods)
        size_ok = test_size == n_slider.value

        # Проверка 2. Все значения присутствуют в исходных данных
        values_ok = np.isin(generated_periods, sorted_periods).all()

        # Проверка 3. Одинаковый Seed даёт одинаковый результат
        test_rng = np.random.default_rng(seed_slider.value)
        test_uniform = test_rng.random(n_slider.value)
        test_indices = np.floor(
            test_uniform * len(sorted_periods)
        ).astype(int)
        repeated_values = sorted_periods[test_indices]

        seed_ok = np.array_equal(generated_periods, repeated_values)

        # Проверка 4. Среднее и медиана близки к исходным
        mean_error = abs(
            np.mean(generated_periods) - np.mean(periods)
        ) / abs(np.mean(periods)) * 100

        median_error = abs(
            np.median(generated_periods) - np.median(periods)
        ) / abs(np.median(periods)) * 100

        statistics_ok = mean_error < 5 and median_error < 5

        results = [
            ("Количество значений", size_ok),
            ("Значения взяты из исходных данных", values_ok),
            ("Повторяемость при одинаковом Seed", seed_ok),
            ("Близость среднего и медианы", statistics_ok)
        ]

        for name, result in results:
            print(("✓ " if result else "✗ ") + name)

        print("\nОтклонение среднего:", round(mean_error, 2), "%")
        print("Отклонение медианы:", round(median_error, 2), "%")

        if all(result for _, result in results):
            print("\nВсе проверки пройдены!")
        else:
            print("\nНекоторые проверки не пройдены.")

    check_generator()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Вывод по проверке генератора

    В ходе тестирования генератора были успешно пройдены все четыре проверки.

    1. Генератор создаёт заданное количество случайных значений.
    2. Все сгенерированные значения присутствуют в исходной выборке.
    3. При использовании одинакового начального значения Seed генератор выдаёт одинаковую последовательность.
    4. Среднее и медиана сгенерированной выборки близки к характеристикам исходных данных. Отклонение среднего составило 2,55%, медианы — 1,53%.

    Таким образом, генератор работает корректно и позволяет воспроизводить эмпирическое распределение периодов обращения экзопланет.
    """)
    return


@app.cell(hide_code=True)
def _(df, np, pd):
    mass_values = pd.to_numeric(
        df["Масса (массы Юпитера)"],
        errors="coerce"
    ).dropna()

    mass_values = mass_values[mass_values > 0].to_numpy()

    sorted_masses = np.sort(mass_values)

    print("Количество планет:", len(sorted_masses))
    print("Минимальная масса:", round(sorted_masses.min(), 6))
    print("Максимальная масса:", round(sorted_masses.max(), 6))
    print("Средняя масса:", round(sorted_masses.mean(), 6))
    return mass_values, sorted_masses


@app.cell(hide_code=True)
def _(n_slider, np, pd, seed_slider, sorted_masses):
    mass_rng = np.random.default_rng(seed_slider.value)

    mass_uniform = mass_rng.random(n_slider.value)

    mass_indices = np.floor(
        mass_uniform * len(sorted_masses)
    ).astype(int)

    generated_masses = sorted_masses[mass_indices]

    print("Генерация масс завершена!")
    print("Количество значений:", len(generated_masses))

    pd.DataFrame({
        "Сгенерированная масса (массы Юпитера)": generated_masses[:10]
    })
    return (generated_masses,)


@app.cell(hide_code=True)
def _(generated_masses, mass_values, np, pd):
    mass_metrics = {
        "Среднее": [
            np.mean(mass_values),
            np.mean(generated_masses)
        ],
        "Медиана": [
            np.median(mass_values),
            np.median(generated_masses)
        ],
        "Дисперсия": [
            np.var(mass_values),
            np.var(generated_masses)
        ],
        "Стандартное отклонение": [
            np.std(mass_values),
            np.std(generated_masses)
        ],
        "Минимум": [
            np.min(mass_values),
            np.min(generated_masses)
        ],
        "Максимум": [
            np.max(mass_values),
            np.max(generated_masses)
        ],
        "25% квантиль": [
            np.quantile(mass_values, 0.25),
            np.quantile(generated_masses, 0.25)
        ],
        "75% квантиль": [
            np.quantile(mass_values, 0.75),
            np.quantile(generated_masses, 0.75)
        ],
        "90% квантиль": [
            np.quantile(mass_values, 0.90),
            np.quantile(generated_masses, 0.90)
        ]
    }

    mass_comparison = pd.DataFrame(
        mass_metrics,
        index=["Реальные данные", "Генератор"]
    ).T

    mass_comparison["Абсолютное отклонение"] = abs(
        mass_comparison["Реальные данные"] -
        mass_comparison["Генератор"]
    )

    mass_comparison["Относительное отклонение, %"] = (
        mass_comparison["Абсолютное отклонение"] /
        mass_comparison["Реальные данные"].abs()
    ) * 100

    mass_comparison.round(4)
    return


@app.cell(hide_code=True)
def _(generated_masses, mass_values, np, plt, sorted_masses):
    def plot_mass_comparison():
        figure, axis = plt.subplots(figsize=(11, 5))

        bins = np.logspace(
            np.log10(sorted_masses.min()),
            np.log10(sorted_masses.max()),
            50
        )

        axis.hist(
            mass_values,
            bins=bins,
            weights=np.ones(len(mass_values)) / len(mass_values),
            alpha=0.5,
            label="Реальные данные",
            color="#91AFFF"
        )

        axis.hist(
            generated_masses,
            bins=bins,
            weights=np.ones(len(generated_masses)) / len(generated_masses),
            alpha=0.5,
            label="Сгенерированные данные",
            color="#72E5C2"
        )

        axis.set_xscale("log")
        axis.set_title("Сравнение реальных и сгенерированных масс экзопланет")
        axis.set_xlabel("Масса (массы Юпитера, логарифмическая шкала)")
        axis.set_ylabel("Доля наблюдений")
        axis.legend()
        axis.grid(alpha=0.25)

        plt.show()

    plot_mass_comparison()
    return


@app.cell(hide_code=True)
def _(generated_masses, mass_values, n_slider, np, seed_slider, sorted_masses):
    def check_mass_generator():
        # Проверка количества значений
        size_ok = len(generated_masses) == n_slider.value

        # Проверка принадлежности исходным данным
        values_ok = np.isin(generated_masses, sorted_masses).all()

        # Проверка повторяемости Seed
        test_rng = np.random.default_rng(seed_slider.value)
        test_uniform = test_rng.random(n_slider.value)

        test_indices = np.floor(
            test_uniform * len(sorted_masses)
        ).astype(int)

        repeated_masses = sorted_masses[test_indices]

        seed_ok = np.array_equal(
            generated_masses,
            repeated_masses
        )

        # Проверка статистических характеристик
        mean_error = abs(
            np.mean(generated_masses) - np.mean(mass_values)
        ) / abs(np.mean(mass_values)) * 100

        median_error = abs(
            np.median(generated_masses) - np.median(mass_values)
        ) / abs(np.median(mass_values)) * 100

        statistics_ok = mean_error < 5 and median_error < 5

        results = [
            ("Количество значений", size_ok),
            ("Значения взяты из исходных данных", values_ok),
            ("Повторяемость при одинаковом Seed", seed_ok),
            ("Близость среднего и медианы", statistics_ok)
        ]

        for name, result in results:
            print(("✓ " if result else "✗ ") + name)

        print("\nОтклонение среднего:", round(mean_error, 2), "%")
        print("Отклонение медианы:", round(median_error, 2), "%")

        if all(result for _, result in results):
            print("\nВсе проверки пройдены!")
        else:
            print("\nНекоторые проверки не пройдены.")

    check_mass_generator()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Общий вывод

    В ходе работы были разработаны два интерактивных генератора случайных величин на основе данных NASA Exoplanet Archive:

    1. Генератор массы экзопланет (в массах Юпитера).
    2. Генератор периода обращения экзопланет (в земных сутках).

    Оба генератора используют метод выборки с возвращением из исходного эмпирического распределения. Пользователь может задавать количество генерируемых значений и начальное значение Seed.

    Для каждого генератора было создано по 100 000 значений. Сравнение статистических характеристик показало небольшие отклонения от исходных данных.

    Для периода обращения отклонение среднего составило 2,55%, медианы — 1,53%. Для массы экзопланет отклонение среднего составило 0,24%, медианы — 0,62%.

    Оба генератора успешно прошли четыре автоматические проверки: проверку количества значений, принадлежности исходной выборке, воспроизводимости результатов при одинаковом Seed и близости статистических характеристик.

    Таким образом, разработанные генераторы корректно воспроизводят эмпирические распределения двух исследуемых случайных величин и позволяют экспериментировать с генерацией данных.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Математическое обоснование генератора

    В работе используется метод генерации случайных значений на основе эмпирического распределения исходной выборки.

    Пусть имеется исходная выборка:

    \[
    X = \{x_1,x_2,\ldots,x_N\},
    \]

    где \(N\) — количество наблюдений.

    Сначала значения упорядочиваются по возрастанию:

    \[
    x_{(1)} \leq x_{(2)} \leq \ldots \leq x_{(N)}.
    \]

    Далее генератор создаёт случайное число \(U\), равномерно распределённое на интервале от 0 до 1:

    \[
    U \sim U(0,1).
    \]

    По нему определяется случайный индекс:

    \[
    J = \lfloor N \cdot U \rfloor + 1.
    \]

    Итоговое сгенерированное значение:

    \[
    X^* = x_{(J)}.
    \]

    Таким образом, каждое наблюдение исходной выборки имеет одинаковую вероятность быть выбранным:

    \[
    P(X^*=x_{(i)})=\frac{1}{N}
    \]

    для каждого отдельного наблюдения. Если в исходных данных встречаются одинаковые значения, вероятность такого значения определяется количеством его повторений.

    Генерация выполняется с возвращением: после выбора значения оно не удаляется из исходной выборки и может быть выбрано повторно.

    При увеличении количества генерируемых значений их статистические характеристики должны приближаться к характеристикам исходной эмпирической выборки. Это объясняется законом больших чисел.

    Метод позволяет воспроизводить эмпирическое распределение исходных данных, однако не создаёт новых значений за пределами наблюдавшейся выборки.
    """)
    return


@app.cell(hide_code=True)
def _(np, pd, plt, sorted_periods):
    def check_convergence():
        sample_sizes = [100, 500, 1000, 5000, 10000, 50000, 100000]

        real_mean = np.mean(sorted_periods)
        errors = []

        for size in sample_sizes:
            rng = np.random.default_rng(42)
            indices = rng.integers(0, len(sorted_periods), size=size)
            sample = sorted_periods[indices]

            error = abs(np.mean(sample) - real_mean) / abs(real_mean) * 100
            errors.append(error)

        figure, axis = plt.subplots(figsize=(10, 5))
        axis.plot(sample_sizes, errors, marker="o")
        axis.set_xscale("log")
        axis.set_title("Сходимость среднего при увеличении выборки")
        axis.set_xlabel("Количество сгенерированных значений")
        axis.set_ylabel("Относительное отклонение среднего, %")
        axis.grid(alpha=0.3)
        plt.show()

        pd.DataFrame({
            "Количество значений": sample_sizes,
            "Отклонение среднего, %": np.round(errors, 4)
        })

    check_convergence()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ограничения генератора

    Использованный метод имеет несколько ограничений:

    1. Генератор выбирает только те значения, которые уже присутствуют в исходном датасете.
    2. Он не создаёт новые значения между наблюдениями и не выходит за пределы исходной выборки.
    3. При генерации с возвращением одно и то же значение может встречаться несколько раз.
    4. Результат зависит от качества и состава исходного датасета. Если в нём присутствуют ошибки или смещения, они сохраняются и в сгенерированных данных.
    5. Генератор воспроизводит эмпирическое распределение, но не гарантирует полного соответствия неизвестному истинному распределению всех экзопланет.

    Таким образом, данный метод подходит для моделирования случайного выбора наблюдений из имеющейся выборки, но не заменяет физическую модель образования и движения экзопланет.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Итоговый вывод

    В ходе работы был реализован интерактивный генератор случайных значений на основе эмпирического распределения данных об экзопланетах.

    Генератор позволяет выбирать количество создаваемых значений и начальное значение Seed, благодаря чему можно управлять процессом генерации и повторять результаты.

    Для проверки работы генератора были сопоставлены статистические характеристики исходных и сгенерированных данных, построены графики распределений и выполнены автоматические тесты. Проверки подтвердили правильность количества генерируемых значений, принадлежность значений исходной выборке и воспроизводимость результата при одинаковом Seed.

    При увеличении размера генерируемой выборки статистические характеристики приближаются к характеристикам исходных данных. Следовательно, реализованный генератор корректно воспроизводит эмпирическое распределение выбранной случайной величины.

    Основное ограничение метода заключается в том, что он использует только уже существующие наблюдения и не создаёт новых значений за пределами исходного датасета.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Использованные источники

    1. NASA Exoplanet Archive. Planetary Systems Composite Parameters. Датасет использован в качестве исходной выборки. https://exoplanetarchive.ipac.caltech.edu/

    2. Efron B., Tibshirani R. J. *An Introduction to the Bootstrap*. Chapman & Hall/CRC, 1993. Книга посвящена методам повторной выборки с возвращением и статистическому моделированию.

    3. NumPy Documentation. Random sampling. https://numpy.org/doc/stable/reference/random/index.html

    4. NumPy Documentation. Generator.integers. https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.integers.html

    5. Marimo Documentation. https://docs.marimo.io/
    """)
    return


if __name__ == "__main__":
    app.run()
