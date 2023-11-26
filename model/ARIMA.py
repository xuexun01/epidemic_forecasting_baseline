import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from scipy.integrate import odeint
from scipy.optimize import minimize


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--learn_rate", type=float, default=0.001)
parser.add_argument("--model", type=str, default="sird")
parser.add_argument("--hist_window", type=int, default=14)
parser.add_argument("--pred_window", type=int, default=14)
args = parser.parse_args()


def get_data(dates, data_path, state):
    cases = []
    for date in dates:
        df = pd.read_csv(f"{data_path}covid_19_daily_reports_us/{date}.csv")
        cases.append(df.loc[df["Province_State"] == state, "Confirmed"].values[0])
    return np.array(cases)



def get_SIRD_data(dates, data_path, state):
    state_info = pd.read_csv(f"{data_path}state_info.csv", index_col=0)
    N = int(state_info.loc[state, "Resident Population"].replace(',', ''))
    susceptible = []
    infected = []
    recovered = []
    decreased = []
    for date in dates:
        df = pd.read_csv(f"{data_path}covid_19_daily_reports_us/{date}.csv", index_col=1).fillna(0)
        confirmed_case = df.loc[state, "Confirmed"]
        recovered_case = df.loc[state, "Recovered"]
        death_case = df.loc[state, "Deaths"]
        active = df.loc[state, "Active"]
        susceptible_case = N - confirmed_case
        recovered_case = confirmed_case - active - death_case if recovered_case == 0 else recovered_case
        
        susceptible.append(susceptible_case)
        infected.append(active)
        recovered.append(recovered_case)
        decreased.append(death_case)
    return susceptible, infected, recovered, decreased


def create_time_series(data, hist_window, pred_window):
    X, y = [], []
    for i in range(len(data) - hist_window - pred_window + 1):
        X.append(data[i:i+hist_window])
        y.append(data[i+hist_window: i+hist_window+pred_window])
    return np.array(X), np.array(y)

# SIRD model
def SIRD(y, t, beta, gamma, delta):
    S, I, R, D = y
    dSdt = -beta * S
    dIdt = beta * S - (gamma + delta) * I
    dRdt = gamma * I
    dDdt = delta * I
    return [dSdt, dIdt, dRdt, dDdt]


def loss_function(params, susceptible, infected, recovered, decreased, initial_conditions):
    beta, gamma, delta = params
    times = np.arange(len(infected))
    ode_solution = odeint(SIRD, initial_conditions, times, args=(beta, gamma, delta))
    loss = np.sum((ode_solution[:, 1] - infected) ** 2) + np.sum((ode_solution[:, 0] - susceptible) ** 2) + np.sum((ode_solution[:, 2] - recovered) ** 2) + np.sum((ode_solution[:, 3] - decreased) ** 2)
    return loss


data_path = "./data/"
pop_data = pd.read_csv(f"{data_path}state_info.csv")
state_list = list(pop_data["State"])



forecasts = []
if args.model == "arima":
    dates = pd.date_range(start="2020-05-01", end="2020-12-01")
    dates = list(dates.strftime("%Y_%m_%d"))
    data = get_data(dates, data_path, state="Alabama")

    train_size = int(len(data) * 0.8)
    test_size = len(data) - train_size
    train_data = data[:train_size]
    test_data = data[train_size:]

    X_train, y_train = create_time_series(train_data, args.hist_window, args.pred_window)
    X_test, y_test = create_time_series(test_data, args.hist_window, args.pred_window)

    total_mse = 0
    for hist, pred in zip(X_test, y_test):
        model = ARIMA(hist, order=(args.hist_window, 1, 1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=args.pred_window)
        forecasts.append(forecast.tolist()[-1])
        mse = mean_squared_error(pred, forecast)/args.pred_window
        print("Test loss (MSE):", mse)
        total_mse += mse
    
    plt.plot(test_data, label='Actual')
    plt.plot([args.hist_window+args.pred_window+i for i in range(len(forecasts))], forecasts, label='Forecast')
    plt.legend()
    plt.show()

    result = total_mse/X_test.shape[0]
    print(f"Total MSE: {result}")


else:
    dates = pd.date_range(start="2020-05-01", end="2021-03-06")
    dates = list(dates.strftime("%Y_%m_%d"))
    # for state in state_list:
    #     susceptible, infected, recovered, decreased = get_SIRD_data(dates, data_path="./data/", state=state)

    susceptible, infected, recovered, decreased = get_SIRD_data(dates, data_path="./data/", state="Alabama")
    train_size = int(len(susceptible) * 0.8)
    test_size = len(susceptible) - train_size
    susc_train_data = susceptible[:train_size]
    susc_test_data = susceptible[train_size:]
    infe_train_data = infected[:train_size]
    infe_test_data = infected[train_size:]
    reco_train_data = recovered[:train_size]
    reco_test_data = recovered[train_size:]
    decr_train_data = decreased[:train_size]
    decr_test_data = decreased[train_size:]

    susc_X_test, susc_y_test = create_time_series(susc_test_data, args.hist_window, args.pred_window)
    infe_X_test, infe_y_test = create_time_series(infe_test_data, args.hist_window, args.pred_window)
    reco_X_test, reco_y_test = create_time_series(reco_test_data, args.hist_window, args.pred_window)
    decr_X_test, decr_y_test = create_time_series(decr_test_data, args.hist_window, args.pred_window)

    forecasts = []
    for s_hist, s_pred, i_hist, i_pred, r_hist, r_pred, d_hist, d_pred in zip(susc_X_test, susc_y_test, infe_X_test, infe_y_test, reco_X_test, reco_y_test, decr_X_test, decr_y_test):
        opt_initial_conditions = [s_hist[0], i_hist[0], r_hist[0], d_hist[0]]
        result = minimize(loss_function, x0=(0.1, 0, 0), args=(s_hist, i_hist, r_hist, d_hist, opt_initial_conditions), method='L-BFGS-B')

        beta_opt, gamma_opt, delta_opt = result.x
        print(f"Optimal parameters: beta={beta_opt:.4f}, gamma={gamma_opt:.4f}, delta={delta_opt:.4f}")

        times = np.arange(len(s_pred)+1)
        initial_conditions = [s_hist[-1], i_hist[-1], r_hist[-1], d_hist[-1]]
        ode_solution_optimal = odeint(SIRD, initial_conditions, times, args=(beta_opt, gamma_opt, delta_opt))
        forecasts.append(ode_solution_optimal[:, 1].tolist()[-1])

    print(forecasts)
    print(infe_test_data)
    # 绘制拟合结果
    plt.plot(np.arange(len(infe_test_data)), infe_test_data, 'r-', label='Actual Infected')
    plt.plot([args.hist_window+args.pred_window+i for i in range(len(forecasts))], forecasts, 'b-', label='Fitted Infected')
    plt.legend()
    plt.xlabel('Days')
    plt.ylabel('Infected')
    plt.title('SIRD Model Fitting')
    plt.show()