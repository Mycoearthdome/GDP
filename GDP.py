import matplotlib.pyplot as plt
import math
import statistics


def amortization_schedule(amount, annual_rate, years):
    """
    Returns list of tuples (interest, principal) for each quarter over the loan term.
    """
    quarters = years * 4
    q_rate = annual_rate / 4
    schedule = []

    if q_rate == 0:
        payment = amount / quarters
        for _ in range(quarters):
            schedule.append((0, round(payment, 4)))
    else:
        payment = amount * q_rate / (1 - (1 + q_rate) ** -quarters)
        remaining = amount
        for _ in range(quarters):
            interest = remaining * q_rate
            principal = payment - interest
            schedule.append((round(interest, 4), round(principal, 4)))
            remaining -= principal

    return schedule


def estimate_canada_proxy_loans(start_year=1905, end_year=2024):
    """
    Estimates Canada's share of federal loans based on population share proxy.
    Returns a list of quarterly Canada-specific new loan amounts.
    """
    def share(year):
        if year < 1950:
            return 0.04 + (year - 1905) * (0.07 - 0.04) / (1950 - 1905)
        elif year < 1980:
            return 0.07 + (year - 1950) * (0.09 - 0.07) / (1980 - 1950)
        elif year < 2000:
            return 0.09 + (year - 1980) * (0.10 - 0.09) / (2000 - 1980)
        elif year <= 2024:
            return 0.10 + (year - 2000) * (0.115 - 0.10) / (2024 - 2000)
        else:
            return 0.115

    federal_annual_base = [0 if y < 1930 else 5 + (y - 1930) * 1 for y in range(start_year, end_year + 1)]
    quarterly_proxy = []

    for i, fed_amt in enumerate(federal_annual_base):
        year = start_year + i
        q_amt = fed_amt / 4
        s = share(year)
        quarterly_proxy.extend([round(q_amt * s, 2)] * 4)

    return quarterly_proxy

def dynamic_loan_growth(i, start_year):
    year = start_year + i // 4
    if 2020 <= year <= 2021:
        return 0.30  # QE era: 30% annual growth
    elif 2022 <= year <= 2024:
        return 0.05  # Tapering
    else:
        return 0.0175  # Long-run normal growth (BOC RATE.)

def adjust_schedule(data, BoC_interest_rate):
    repayments = amortization_schedule(data["Balance End (CAD bn)"], BoC_interest_rate, 1)
    total_interest = 0
    total_principal = 0
    for quarter in repayments:
        total_interest += quarter[0]
        total_principal += quarter[1]
    data["Interest (CAD bn)"] = total_interest
    data["BoC Principal (CAD bn)"] = total_principal
    return data



def generate_federal_debt_schedule_predictive(
    initial_debt,
    annual_interest_rate,
    quarterly_new_loans,
    start_year,
    years_to_predict=5,
    loan_growth_rate=0.0,
    crisis_loans=None,
):
    # Support dynamic loan growth as callable or dict
    if not callable(loan_growth_rate):
        fixed_growth = loan_growth_rate
        loan_growth_rate = lambda i: fixed_growth

    crisis_loans = crisis_loans or {}
    balance = initial_debt
    schedule = []
    total_interest_paid = 0.0
    quarterly_rate = annual_interest_rate / 4
    total_quarters = len(quarterly_new_loans)
    avg_loan = sum(quarterly_new_loans) / len(quarterly_new_loans) if quarterly_new_loans else 0
    boc_repayments = []  # List of (start_index, [(interest, principal), ...])

    for i in range(1, total_quarters + 1):
        year = start_year + (i - 1) // 4
        quarter_num = (i - 1) % 4 + 1
        q_label = f"Q{quarter_num}"

        if i <= len(quarterly_new_loans):
            new_loan = quarterly_new_loans[i - 1]
        else:
            growth_periods = i - len(quarterly_new_loans)
            growth_factor = (1 + loan_growth_rate(i)) ** growth_periods
            new_loan = avg_loan * growth_factor

        injection = 0
        boc_interest_this_q = 0
        boc_principal_this_q = 0

        if (year, q_label) in crisis_loans:
            loan = crisis_loans[(year, q_label)]
            injection = loan["amount"]
            repayments = amortization_schedule(loan["amount"], loan["interest_rate"], loan["term_years"])
            boc_repayments.append((i, repayments))

        for start_index, rep_list in boc_repayments:
            offset = i - start_index
            if 0 <= offset < len(rep_list):
                interest, principal = rep_list[offset]
                boc_interest_this_q += interest
                boc_principal_this_q += principal

        interest = balance * quarterly_rate
        total_interest_paid += interest
        total_added = new_loan + injection
        balance += interest + total_added - boc_principal_this_q

        schedule.append({
            "Year": year,
            "Quarter": q_label,
            "Interest (CAD bn)": round(interest, 3),
            "New Loans (CAD bn)": round(new_loan, 3),
            "Crisis Injection (CAD bn)": round(injection, 3),
            "BoC Interest (CAD bn)": round(boc_interest_this_q, 3),
            "BoC Principal (CAD bn)": round(boc_principal_this_q, 3),
            "Total Added (CAD bn)": round(total_added, 3),
            "Balance End (CAD bn)": round(balance, 3)
        })

    return schedule, round(total_interest_paid, 3)


def find_debt_gdp_crossover_quarterly(schedule, this_year, initial_gdp, quarterly_gdp_growth_series):
    """
    Find the first quarter when federal debt exceeds GDP using quarterly GDP growth series.
    """
    gdp = initial_gdp
    for i, entry in enumerate(schedule):
        debt = entry["Balance End (CAD bn)"]
        if debt > gdp and entry["Year"] >= this_year:
            return (entry["Year"], entry["Quarter"], debt, gdp)
        if i < len(quarterly_gdp_growth_series):
            gdp = quarterly_gdp_growth_series[i]
        else:
            # If out of quarterly growth data, assume last known growth or no growth
            gdp *= 1

    return None


def print_quarterly_schedule(schedule, outstanding_interest_balance, max_rows=20):
    headers = ["Year", "Quarter", "Interest", "New Loans", "Crisis", "BoC Interest", "BoC Principal", "Balance End"]
    widths = [6, 8, 14, 12, 10, 14, 15, 16]
    sep = "+" + "+".join("-" * w for w in widths) + "+"

    print(sep)
    print("|" + "|".join(h.center(w) for h, w in zip(headers, widths)) + "|")
    print(sep)

    for row in schedule[:max_rows]:
        print("|" +
              str(row["Year"]).rjust(widths[0]) + "|" +
              row["Quarter"].center(widths[1]) + "|" +
              f"{row['Interest (CAD bn)']:,.3f}".rjust(widths[2]) + "|" +
              f"{row['New Loans (CAD bn)']:,.3f}".rjust(widths[3]) + "|" +
              f"{row['Crisis Injection (CAD bn)']:,.3f}".rjust(widths[4]) + "|" +
              f"{row['BoC Interest (CAD bn)']:,.3f}".rjust(widths[5]) + "|" +
              f"{row['BoC Principal (CAD bn)']:,.3f}".rjust(widths[6]) + "|" +
              f"{row['Balance End (CAD bn)']:,.3f}".rjust(widths[7]) + "|")

    print(sep)
    if len(schedule) > max_rows:
        print(f"... ({len(schedule) - max_rows} more quarters)")
    print(f"Outstanding interest balance: CAD {outstanding_interest_balance:,.3f} Billion")


def project_gdp_series(initial_gdp, annual_growth, num_quarters):
    gdp = initial_gdp
    quarterly_growth = (1 + annual_growth) ** 0.25 - 1
    gdp_series = []
    for _ in range(num_quarters):
        gdp_series.append(gdp)
        gdp *= (1 + quarterly_growth)
    return gdp_series


def evaluate_fit(balances, gdp_series, method="mse"):
    if method == "mse":
        errors = [(d - g) ** 2 for d, g in zip(balances, gdp_series)]
        return sum(errors) / len(errors)
    elif method == "log_mse":
        errors = [(math.log(d + 1) - math.log(g + 1)) ** 2 for d, g in zip(balances, gdp_series)]
        return sum(errors) / len(errors)
    elif method == "ratio_stdev":
        ratios = [d / g for d, g in zip(balances, gdp_series)]
        return statistics.stdev(ratios)
    else:
        raise ValueError(f"Unknown fit method: {method}")


def find_best_gdp_growth(
    balances,
    initial_gdp,
    min_rate=0.002,
    max_rate=0.1,
    step=0.0005,
    method="mse"
):
    best_rate = None
    best_fit = float('inf')
    rate = min_rate

    while rate <= max_rate:
        gdp_series = project_gdp_series(initial_gdp, rate, len(balances))
        fit = evaluate_fit(balances, gdp_series, method=method)

        if fit < best_fit:
            best_fit = fit
            best_rate = rate

        rate += step

    return best_rate, best_fit


def estimate_exponential_growth_rate(values):
    start = values[0]
    end = values[-1]
    periods = len(values) - 1
    return (end / start) ** (1 / periods) - 1


def find_unsustainable_tax_quarter_quarterly(schedule, initial_gdp, quarterly_gdp_growth_series, Boc_Sums_Collected, tax_to_gdp_ratio=0.20):
    """
    Find fiscal unsustainability quarter using quarterly GDP growth series.
    """
    gdp = initial_gdp
    tax_ratio_quarterly = tax_to_gdp_ratio / 4

    for i, entry in enumerate(schedule):
        interest = entry["Interest (CAD bn)"]
        principal = entry["BoC Principal (CAD bn)"]
        required_payment = interest + principal
        revenue = tax_ratio_quarterly * gdp

        if i < len(quarterly_gdp_growth_series):
            gdp = quarterly_gdp_growth_series[i]
        else:
            gdp *= 1  # No growth assumed beyond data

        if required_payment > revenue + Boc_Sums_Collected:
            return {
               "Year": entry["Year"],
               "Quarter": entry["Quarter"],
               "Required Payment": required_payment,
               "Tax Revenue": revenue,
               "Debt": entry["Balance End (CAD bn)"],
               "GDP": gdp
            }, Boc_Sums_Collected-required_payment
        
    return None, Boc_Sums_Collected-required_payment

def plotting(
    initial_gdp,
    this_year,
    label_extension,
    real_gdp_quarterly_growth,
    tax_to_gdp_ratio,
    gdp_growth_rate,
    schedule_covid=None,
    outstanding_interest_balance=None,
    BoC_Sums_Collected=None
):

    print_quarterly_schedule(schedule_covid, outstanding_interest_balance, max_rows=len(schedule_covid))

    crossover = find_debt_gdp_crossover_quarterly(schedule_covid, this_year, initial_gdp, real_gdp_quarterly_growth)
    if crossover:
        y, q, d, g = crossover
        print(f"\n📉 Crossover Point: Debt surpasses GDP in {y} {q}")
        print(f"Debt: CAD {d:,.3f}")
        print(f"GDP: CAD {g:,.3f}")
    else:
        print("\n✅ Debt does not surpass GDP within projection window.")

    years_covid = [f"{entry['Year']} {entry['Quarter']}" for entry in schedule_covid]
    balances_covid = [entry["Balance End (CAD bn)"] for entry in schedule_covid]

    unsustainable, BoC_Sums_Remaining = find_unsustainable_tax_quarter_quarterly(schedule_covid, initial_gdp, real_gdp_quarterly_growth, BoC_Sums_Collected, tax_to_gdp_ratio=tax_to_gdp_ratio, )
    if unsustainable:
        print("\n🚨 Fiscal Unsustainability Detected")
        print(f"Quarter: {unsustainable['Year']} {unsustainable['Quarter']}")
        print(f"Required Debt Payment: CAD {unsustainable['Required Payment']:.2f} Billion")
        print(f"Available Tax Revenue: CAD {unsustainable['Tax Revenue']:.2f} Billion")
        print(f"Debt: CAD {unsustainable['Debt']:.2f} Billion")
        print(f"GDP: CAD {unsustainable['GDP']:.2f} Billion")
        print(f"Debt:GDP ratio = {unsustainable['Debt'] / unsustainable['GDP'] * 100:.2f} %")
    else:
        print("\n✅ Taxes can cover debt payments within projection window.")

    debt_growth_q = estimate_exponential_growth_rate(balances_covid)
    debt_growth_a = (1 + debt_growth_q) ** 4 - 1

    print(f"\n🔍 Estimated Quarterly Debt Growth Rate: {debt_growth_q:.4%}")
    print(f"📆 Estimated Annualized Debt Growth Rate: {debt_growth_a:.4%}")
    if unsustainable:
        print(f"TRY INCREASING GROWTH ABOVE ESTIMATED ANNUALIZED DEBT GROWTH RATE :D --> in main --> gdp_growth = {gdp_growth_rate} < {debt_growth_a*100:.4}")

    # Plot
    plt.figure(figsize=(16, 6))

    plt.plot(years_covid, balances_covid, label="Federal Debt - COVID Scenario", color="crimson")
    plt.plot(years_covid, real_gdp_quarterly_growth, label="GDP (Projected - Real Quarterly Growth)", color="green", linestyle="-.")
    # Optionally, could plot a best fit GDP as well if needed

    if unsustainable:
        label = f"{unsustainable['Year']} {unsustainable['Quarter']}"
        plt.axvline(
            x=label,
            color='red',
            linestyle='-',
            linewidth=2.5,
            alpha=0.6,
            label='Unsustainable Fiscal Point',
            zorder=1
        )
    plt.title(f"Canada Federal Debt and GDP Over Time - {label_extension} projection")
    plt.xlabel("Year & Quarter")
    plt.ylabel("CAD (billion)")
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    if unsustainable:
        return unsustainable['Debt'] / unsustainable['GDP'] * 100, unsustainable["Year"], BoC_Sums_Remaining
    else:
        # Return estimated debt growth compared to GDP end value
        return debt_growth_a / real_gdp_quarterly_growth[-1]*4 * 100, None, BoC_Sums_Remaining

def generate_quarterly_inflation_series(historical_data, multiplier):
    """
    Generates a list of quarterly inflation rates (as decimals) based on annual inflation rates.
    Assumes quarterly compounding.
    """
    quarterly_inflation_rates = []
    for year_data in historical_data:
        annual_inflation_pct = year_data["inflation"]
        annual_rate = annual_inflation_pct / 100
        quarterly_rate = (1 + annual_rate) ** (1/4) - 1
        quarterly_rate = quarterly_rate * multiplier
        quarterly_inflation_rates.extend([quarterly_rate] * 4)
    return quarterly_inflation_rates

def compute_real_gdp_per_capita_growth(data, initial_gdp, initial_population=41.5):  # e.g., 5.37 million in 1905 41.5 Million in 2025
    results = []
    current_gdp = initial_gdp
    current_pop = initial_population

    for entry in data:
        year = entry["year"]
        nominal = entry["nominal_gdp_growth"]
        inflation = entry["inflation"]
        pop_growth = entry["population_growth"]

        real_gdp = nominal - inflation
        real_gdp_per_capita_growth = real_gdp - pop_growth

        growth_factor = 1 + real_gdp_per_capita_growth / 100
        pop_growth_factor = 1 + pop_growth / 100

        current_gdp *= growth_factor
        current_pop *= pop_growth_factor
        total_real_gdp = current_gdp

        quarterly_growth_factor = growth_factor ** (1 / 4)
        quarterly_gdps = [current_gdp * (quarterly_growth_factor ** i) for i in range(1, 5)]

        results.append({
            "year": year,
            "real_gdp_per_capita_growth": real_gdp_per_capita_growth,
            "real_gdp_per_capita": current_gdp,
            "population": current_pop,
            "total_real_gdp": total_real_gdp,
            "quarterly_real_gdp": quarterly_gdps
        })

    return results



def generate_canada_historical_data(years_projection, gdp_growth, start_year):
    # Approximate population in millions for census years
    pop_census = {
        1901: 5.37,
        1911: 7.21,
        1921: 8.79,
        1931: 10.38,
        1941: 11.51,
        1951: 14.01,
    }

    def yearly_population_growth(prev_pop, next_pop, years):
        growth_rate = ((next_pop / prev_pop) ** (1/years) - 1) * 100
        return growth_rate

    # Calculate population growth for years between census years
    pop_growth_by_year = {}
    census_years = sorted(pop_census.keys())
    for i in range(len(census_years) - 1):
        start = census_years[i]
        end = census_years[i+1]
        years = end - start
        growth = yearly_population_growth(pop_census[start], pop_census[end], years)
        for y in range(start + 1, end + 1):
            pop_growth_by_year[y] = growth

    # For 1952–1960, estimate growth ~2% (post-war boom)
    for y in range(1952, 1961):
        pop_growth_by_year[y] = 2.0

    # For 1961–2023, use known approximations (simplified)
    known_pop_growth = {
        1961: 2.1, 1962: 2.05, 1963: 2.0, 1964: 1.95, 1965: 1.9,
        1970: 1.9, 1975: 1.5, 1980: 1.2, 1985: 1.1, 1990: 1.5,
        1995: 1.1, 2000: 0.9, 2005: 0.8, 2010: 1.1, 2015: 1.0,
        2020: 1.0, 2021: 0.7, 2022: 0.78, 2023: 3.2,
    }

    def interpolate_yearly_growth(start_year, end_year, start_value, end_value):
        years = end_year - start_year
        step = (end_value - start_value) / years
        return [start_value + step * i for i in range(years + 1)]

    pop_years = sorted(known_pop_growth.keys())
    for i in range(len(pop_years) - 1):
        sy, ey = pop_years[i], pop_years[i + 1]
        vals = interpolate_yearly_growth(sy, ey, known_pop_growth[sy], known_pop_growth[ey])
        for offset, val in enumerate(vals[:-1]):
            pop_growth_by_year[sy + offset] = val

    # Nominal GDP and inflation data dictionaries
    nominal_gdp_by_year = {}
    inflation_by_year = {}

    # Fill rough data for early years (1905-1950)
    for y in range(1905, 1951):
        nominal_gdp_by_year[y] = 3.0 if y % 5 != 0 else 2.0  # arbitrary pattern
        inflation_by_year[y] = 2.0 if y < 1920 else 5.0

    # Deflation during Great Depression and post-depression inflation
    for y in range(1920, 1935):
        inflation_by_year[y] = -2.0
    for y in range(1935, 1951):
        inflation_by_year[y] = 4.0

    # More accurate data 1951-2023 (partial)
    nominal_gdp_by_year.update({
        1951: 5.5, 1952: 6.0, 1953: 6.2, 1954: 5.0, 1955: 5.7,
        1960: 4.8, 1965: 5.2, 1970: 5.1, 1975: 3.0, 1980: 12.7,
        1985: 3.1, 1990: 5.1, 1995: 2.5, 2000: 9.8, 2005: 2.7,
        2010: 17.7, 2015: 2.2,
        # Updated recent years:
        2020: -5.0,
        2021: 21.3,
        2022: 7.7,
        2023: 1.5,
        2024: 1.6,
    })

    inflation_by_year.update({
        1951: 10.4, 1955: 3.4, 1960: 1.3, 1965: 3.1, 1970: 3.0,
        1975: 9.1, 1980: 10.0, 1985: 4.4, 1990: 4.8, 1995: 2.2,
        2000: 2.7, 2005: 2.2, 2010: 1.8, 2015: 1.1, 2020: 0.7,
        2021: 3.4, 2022: 6.8, 2023: 3.9,
    })

    def fill_missing(data_dict, start, end):
        keys = list(range(start, end + 1))
        for i in range(len(keys) - 1):
            k1, k2 = keys[i], keys[i + 1]
            if k1 in data_dict and k2 in data_dict:
                diff = data_dict[k2] - data_dict[k1]
                steps = k2 - k1
                for j in range(1, steps):
                    data_dict[k1 + j] = data_dict[k1] + diff * j / steps

    # Fill missing values by linear interpolation
    intervals = [
        (1951, 1960), (1960, 1965), (1965, 1970), (1970, 1975),
        (1975, 1980), (1980, 1985), (1985, 1990), (1990, 1995),
        (1995, 2000), (2000, 2005), (2005, 2010), (2010, 2015),
        (2015, 2020), (2020, 2023)
    ]

    for start, end in intervals:
        fill_missing(nominal_gdp_by_year, start, end)
        fill_missing(inflation_by_year, start, end)

    # Build the full historical data list
    historical_data = []
    for year in range(start_year, 2026+years_projection):
        if year < this_year:
            data = {
                "year": year,
                "nominal_gdp_growth": nominal_gdp_by_year.get(year, 1.4),
                "inflation": round(inflation_by_year.get(year, 1.75+0.25), 2), # Defaults to BoC Rate + 0.25 base points
                "population_growth": round(pop_growth_by_year.get(year, 0.2), 2), #0.2% increase by default. (~ 83000 in 2025 over 41 500 000 habitants)
            }
        else:
            data = {
                "year": year,
                "nominal_gdp_growth": nominal_gdp_by_year.get(year, gdp_growth),
                "inflation": round(inflation_by_year.get(year, 1.75+0.25), 2), # Defaults to BoC Rate + 0.25 base points
                "population_growth": round(pop_growth_by_year.get(year, 0.2), 2), #0.2% increase by default. (~ 83000 in 2025 over 41 500 000 habitants)
            }
        historical_data.append(data)

    return historical_data

def Federal_Debt_Acquisition_Boc_DEbt_Repayment(historical_data, crisis_loans,this_year, start_year, years_to_predict, initial_debt,  annual_rate, BoC_Investments_ROI, share_guarantees_tax):

    Base_year = start_year

    BoC_Rate = 0.0175 # 1.75 percent

    quarterly_inflation = generate_quarterly_inflation_series(historical_data, 1)

    yearly_real_gdp = compute_real_gdp_per_capita_growth(historical_data, initial_gdp)

    real_gdp_list = [g for entry in yearly_real_gdp for g in entry["quarterly_real_gdp"]]

    Sums_Collected = 0
    total_interest = 0
    schedule = None
    Count = 1
    loaned = False
    for i, real_gdp in enumerate(real_gdp_list):
        data_index = 0
        if Count == 5:
            Base_year += 1
            if loaned:
                Sums_Collected += Debt_Loan * (1+ BoC_Investments_ROI)
            Count = 1
            loaned = False
        #if Base_year >= this_year:
        if quarterly_inflation[i]*4 >= BoC_Rate:
            Sums_Collected += real_gdp * (share_guarantees_tax - BoC_Rate) # 18% - 1.175% on ressources guarantees behind 467,468.469 of laws of the banks. (18% to keep AAA Rating)
        else:
            #BoC Borrows from the DEBT
            Debt_Loan =  real_gdp * (share_guarantees_tax - BoC_Rate) 
            Sums_Collected += Debt_Loan
            key = (Base_year, f"Q{Count}")
            if key not in crisis_loans:
                crisis_loans.update({key:{"amount": Debt_Loan, "term_years": 15, "interest_rate": 0.0175}}) # 15 years loans ??
            else:
                crisis_loans.update({key:{"amount": crisis_loans[key]["amount"] + Debt_Loan, "term_years": crisis_loans[key]["term_years"], "interest_rate": crisis_loans[key]["interest_rate"]}})
            loaned = True
        Count += 1

    quarterly_new_loans = estimate_canada_proxy_loans(start_year=start_year, end_year=this_year + years_to_predict)

    schedule, total_interest = generate_federal_debt_schedule_predictive(
        initial_debt,
        annual_rate,
        quarterly_new_loans,
        start_year,
        years_to_predict,
        loan_growth_rate=dynamic_loan_growth,
        crisis_loans=crisis_loans,
    )
    reconcile = 0
    for data in schedule:
        if data["Balance End (CAD bn)"] <= Sums_Collected:
            schedule[data_index].update({"Balance End (CAD bn)": 1})  # debt erased to 1 Billion
            Sums_Collected -= data["Balance End (CAD bn)"] - 1
            reconcile = Sums_Collected
        else:
            schedule[data_index].update({"Balance End (CAD bn)": data["Balance End (CAD bn)"]-reconcile})  # debt reduced

        schedule[data_index].update(adjust_schedule(schedule[data_index],BoC_Rate)) #adjusts interest and principal
        data_index += 1

    
    if Sums_Collected >= total_interest:
        Sums_Collected -= total_interest
        total_interest = 0
    else:
        total_interest -= Sums_Collected
        Sums_Collected = 0
    
    return Sums_Collected, crisis_loans, schedule, total_interest

if __name__ == "__main__":
    confederation_year = 1867
    this_year = 2025
    start_year = 2025
    initial_debt = 1800    # 275 million in 1905 0.275 billion (default)
    annual_rate = 0.028
    initial_gdp = 2240    # 169 Billion in 1905 and 2240 for 2025 (2.24 trillion)
    gdp_growth = 1.6     # %%%%%%%%%%%%%%%%%%%%%%%%% 2025 %%%%%%%%%%%%%%%%%%%%%%% --> adjust this to today GDP growth percentage (%).
    corporate_tax = 0.2  # Corporate taxes
    BoC_Sums_Collected = 0
    crisis_loans = {
        (2020, "Q2"): {"amount": 600, "term_years": 15, "interest_rate": 0.0175},  # 1.75% BoC rate
    }
    BoC_Investments_ROI = 0.08 # 8 percent return on investments (Medium Risk)
    share_guarantees_tax = 0.14 # 14% - 1.75% on ressources guarantees behind 467,468.469 of laws of the banks. (18% to keep AAA Rating)
    schedule = None
    total_interest = 0

    years_to_predict_1 = 1

    historical_data = generate_canada_historical_data(years_to_predict_1, gdp_growth, start_year)

    BoC_Sums_Collected, crisis_loans, schedule, total_interest = Federal_Debt_Acquisition_Boc_DEbt_Repayment(historical_data, crisis_loans,this_year, start_year, years_to_predict_1, initial_debt,  annual_rate, BoC_Investments_ROI, share_guarantees_tax) #Comment out to filter out this scenario.

    yearly_real_gdp = compute_real_gdp_per_capita_growth(historical_data, initial_gdp)

    real_gdp_2025_2026_Q1_Q4 = [g for entry in yearly_real_gdp for g in entry["quarterly_real_gdp"]]
                                                                              
    ratio_50 , unsustainable, BoC_Sums_Remaining = plotting(initial_gdp, this_year, f"{years_to_predict_1} years",real_gdp_2025_2026_Q1_Q4, corporate_tax, gdp_growth, schedule, total_interest, BoC_Sums_Collected)

    if not unsustainable:
        years_to_predict_300 = 4 # 300 to find out something about 1.6 % growth (2025)

        historical_data = generate_canada_historical_data(years_to_predict_300, gdp_growth, start_year)

        BoC_Sums_Collected, crisis_loans, schedule, total_interest = Federal_Debt_Acquisition_Boc_DEbt_Repayment(historical_data, crisis_loans,this_year, start_year, years_to_predict_300, initial_debt,  annual_rate, BoC_Investments_ROI, share_guarantees_tax) #Comment out to filter out this scenario.

        yearly_real_gdp = compute_real_gdp_per_capita_growth(historical_data, initial_gdp)

        real_gdp_2025_2325_Q1_Q4 = [g for entry in yearly_real_gdp for g in entry["quarterly_real_gdp"]]

        ratio_300, unsustainable, BoC_Sums_Remaining= plotting(initial_gdp, this_year, f"{years_to_predict_300} years",real_gdp_2025_2325_Q1_Q4, corporate_tax, gdp_growth, schedule, total_interest, BoC_Sums_Collected)

        if unsustainable:
            print(f"RESULT: At year {unsustainable-confederation_year} after Confederation of Canada, taxes alone will not support repayment of the federal debt!")
            print(f"This leaves us...{unsustainable-this_year} years to react.")
        else:
            print("                     _ooOoo_")
            print("                    o8888888o")
            print("                    88\" . \"88")
            print("                    (| -_- |)")
            print("                    O\\  =  /O")
            print("                 ____/`---'\\____")
            print("               .'  \\\\|     |//  `.")
            print("              /  \\\\|||  :  |||//  \\")
            print("             /  _||||| -:- |||||_  \\")
            print("            |   | \\\\\\  -  /'| |    |")
            print("            | \\_|  `\\`---'//  |_/  |")
            print("            \\  .-\\__ `-. -'__/-.  /")
            print("          ___`. .'  /--.--\\  `. .'___")
            print("       .\"\" '<  `.___\\_<|>_/___.' _> \"\".")
            print("      | | :  `- \\`. ;`. _/; .'/ /  .' ; |")
            print("      \\  \\ `-.   \\_\\_`. _.'_/_/  -' _.' /")
            print("       books open O  |_____|  O     /__\\")
            print("       ________________|________________")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("! A L L  F E D E R A L  D E B T  P A I D  I N  F U L L !")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!!!!!!!!!!!!! F I S C A L  P O L I C Y !!!!!!!!!!!!!!!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!!!!!!!!!!!!!-== UNDER EMERGENCY ACT==-!!!!!!!!!!!!!!!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!!!!!!!!!!!!!! --> FREED CASH FLOW <-- !!!!!!!!!!!!!!!!")
            print(f"!!!!!!!!!!!!!!! --> {BoC_Sums_Remaining:.2f} Billion <-- !!!!!!!!!!!!!!!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("! A L L  F E D E R A L  D E B T  P A I D  I N  F U L L !")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        print(f"RESULT: At year {unsustainable-confederation_year} after Confederation of Canada, taxes alone will not support repayment of the federal debt!")
        print(f"This leaves us...{unsustainable-this_year} years to react.")