# Coordinate descent
# Grad 5 gjør det lett å se konvergens; prøv senere f.eks. grad 10 og 15.
degree_e = 5 # beste grad er 8
lmb_e = 0.01 # best lambda
X = design_matrix(x, degree_e)  # Bygg matrisen for graden valgt her.
Xe_train, Xe_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=2026
)
# Skalering læres KUN fra treningssettet. Konstantkolonnen beholdes som 1.
mean_e, std_e = Xe_train.mean(axis=0), Xe_train.std(axis=0)
mean_e[0], std_e[0] = 0.0, 1.0
X_train = (Xe_train - mean_e) / std_e
X_test = (Xe_test - mean_e) / std_e

# What does AD say about d|theta|/dtheta at theta = 0?  (Discuss whether this is a valid subgradient.)
print("jax.grad(jnp.abs)(0.0) =", jax.grad(jnp.abs)(0.0))
print("jax.grad(jnp.abs)(-0.3) =", jax.grad(jnp.abs)(-0.3), " jax.grad(jnp.abs)(0.3) =", jax.grad(jnp.abs)(0.3))
# At zero, abs has no ordinary derivative.
# JAX returns 1.0, which is a valid subgradient in [-1, 1].

# Your Lasso code for part g) here
def soft_treshold(z, gamma):
    return np.sign(z) * np.maximum(np.abs(z) - gamma, 0.0)

def lasso_coord_descent(X, y, lmbda, n_iter=10000, tol=1e-8): # Copy book
    # Colms of X centred and stadardised
    # No intercept is penalised

    n, p = X.shape
    theta = np.zeros(p)
    col_norms = np.sum(X**2, axis=0)
    r = y - X @ theta # full residual

    for _ in range(n_iter):
        theta_old = theta.copy()
        for j in range(p):
            # partial residual: add back the contr. of colmn j
            r += X[:, j] * theta[j]
            rho = X[:, j] @ r
            if j == 0:
                theta[j] = rho / col_norms[j]
            else:
                theta[j] = soft_treshold(rho, lmbda * n/ 2.0) / col_norms[j]
            r -= X[: ,j] * theta[j]
        if np.max(np.abs(theta - theta_old)) < tol:
            break
    return theta

CF = np.linalg.pinv(X_train) @ y_train 
y_cf = X_test @ CF
print(" -- Closed Form --")
print("Cf", CF)

# lambda = 2*alpha
lmb_l = 0.01
theta_lasso = lasso_coord_descent(X_train, y_train, lmb_l)
y_pred_l = X_test @ theta_lasso 
MSE_l = np.mean((y_test - y_pred_l)**2)





# alpha = lambda/2
model = Lasso(alpha=lmb_l/2, max_iter=100000)
# Fjern konstantkolonnen; Lasso håndterer intercept selv.
model.fit(X_train[:, 1:], y_train)
y_pred = model.predict(X_test[:, 1:])

print(lmb_l)

print(" -- Manual Lasso --")
print("theta coeff ", theta_lasso)
print("MSE prediction: ",MSE_l)

print("-- Schikit --")
print("Koeffisienter:", model.coef_)
print("Konstantledd:", model.intercept_)
print("MSE prediction:", np.mean((y_test - y_pred)**2))

print("-- Difference MSE --")
diff = np.mean((y_pred - y_pred_l)**2)
print("MSE Skikit VS Manual", diff)
print("MSE CF vs Manual", np.mean((y_cf - y_pred_l))**2)

print("-----------------------")


