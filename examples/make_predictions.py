import ecnet

server = ecnet.server.Server()
server.create_save_env()
server.import_data()
server.create_mlp_model()
server.fit_mlp_model_validation()
server.select_best()

rmse = server.test_model_rmse()
mae = server.test_model_mae()
r2 = server.test_model_r2()

results = server.use_mlp_model_all()
server.output_results(results, "all", "results_output.csv")

print(rmse)
print(mae)
print(r2)
