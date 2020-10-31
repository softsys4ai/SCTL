# data generation.
LV3 = c("a", "b", "c")

F = sample(LV3, 5000, prob = rep(1/3, 3), replace = TRUE)
G = sample(LV3, 5000, prob = c(0.75, 0.2, 0.05), replace = TRUE)
H = sample(LV3, 5000, prob = c(0.6, 0.2, 0.2), replace = TRUE)
K = sample(LV3, 5000, prob = c(0.15, 0.25, 0.6), replace = TRUE)

U = sample(c("a", "b"), 5000, prob = rep(1/2, 2), replace = TRUE)

C1 = U
C1[C1 == "a"] = sample(LV3, length(which(C1 == "a")), prob = c(0.8, 0.1, 0.1), replace = TRUE)
C1[C1 == "b"] = sample(LV3, length(which(C1 == "b")), prob = c(0.4, 0.2, 0.4), replace = TRUE)

C2 = U
C2[C2 == "a"] = sample(LV3, length(which(C2 == "a")), prob = c(0.67, 0.13, 0.2), replace = TRUE)
C2[C2 == "b"] = sample(LV3, length(which(C2 == "b")), prob = c(0.5, 0.2, 0.3), replace = TRUE)

D = F
D[D == "a"] = sample(LV3, length(which(D == "a")), prob = c(0.4, 0.3, 0.3), replace = TRUE)
D[D == "b"] = sample(LV3, length(which(D == "b")), prob = c(0.2, 0.4, 0.4), replace = TRUE)
D[D == "c"] = sample(LV3, length(which(D == "c")), prob = c(0.1, 0.8, 0.1), replace = TRUE)

E = apply(cbind(G,H), 1, paste, collapse= ":")
E[E == "a:a"] = sample(LV3, length(which(E == "a:a")), prob = c(0.8, 0.1, 0.1), replace = TRUE)
E[E == "a:b"] = sample(LV3, length(which(E == "a:b")), prob = c(0.2, 0.1, 0.7), replace = TRUE)
E[E == "a:c"] = sample(LV3, length(which(E == "a:c")), prob = c(0.4, 0.2, 0.4), replace = TRUE)
E[E == "b:a"] = sample(LV3, length(which(E == "b:a")), prob = c(0.1, 0.8, 0.1), replace = TRUE)
E[E == "b:b"] = sample(LV3, length(which(E == "b:b")), prob = c(0.9, 0.05, 0.05), replace = TRUE)
E[E == "b:c"] = sample(LV3, length(which(E == "b:c")), prob = c(0.3, 0.4, 0.3), replace = TRUE)
E[E == "c:a"] = sample(LV3, length(which(E == "c:a")), prob = c(0.1, 0.1, 0.8), replace = TRUE)
E[E == "c:b"] = sample(LV3, length(which(E == "c:b")), prob = c(0.25, 0.5, 0.25), replace = TRUE)
E[E == "c:c"] = sample(LV3, length(which(E == "c:c")), prob = c(0.15, 0.45, 0.4), replace = TRUE)

I = E
I[I == "a"] = sample(LV3, length(which(I == "a")), prob = c(0.3, 0.4, 0.3), replace = TRUE)
I[I == "b"] = sample(LV3, length(which(I == "b")), prob = c(0.2, 0.4, 0.4), replace = TRUE)
I[I == "c"] = sample(LV3, length(which(I == "c")), prob = c(0.1, 0.8, 0.1), replace = TRUE)

B = apply(cbind(D, C2), 1, paste, collapse= ":")
B[B == "a:a"] = sample(LV3, length(which(B == "a:a")), prob = c(0.8, 0.1, 0.1), replace = TRUE)
B[B == "a:b"] = sample(LV3, length(which(B == "a:b")), prob = c(0.4, 0.5, 0.1), replace = TRUE)
B[B == "b:a"] = sample(LV3, length(which(B == "b:a")), prob = c(0.2, 0.2, 0.6), replace = TRUE)
B[B == "b:b"] = sample(LV3, length(which(B == "b:b")), prob = c(0.3, 0.4, 0.3), replace = TRUE)
B[B == "c:a"] = sample(LV3, length(which(B == "c:a")), prob = c(0.1, 0.1, 0.8), replace = TRUE)
B[B == "c:b"] = sample(LV3, length(which(B == "c:b")), prob = c(0.25, 0.5, 0.25), replace = TRUE)

L = K
L[L == "a"] = sample(LV3, length(which(L == "a")), prob = c(0.2, 0.7, 0.1), replace = TRUE)
L[L == "b"] = sample(LV3, length(which(L == "b")), prob = c(0.2, 0.1, 0.7), replace = TRUE)
L[L == "c"] = sample(LV3, length(which(L == "c")), prob = c(0.4, 0.3, 0.2), replace = TRUE)

J = K
J[J == "a"] = sample(LV3, length(which(J == "a")), prob = c(0.2, 0.4, 0.4), replace = TRUE)
J[J == "b"] = sample(LV3, length(which(J == "b")), prob = c(0.3, 0.4, 0.2), replace = TRUE)
J[J == "c"] = sample(LV3, length(which(J == "c")), prob = c(0.1, 0.8, 0.1), replace = TRUE)

M = apply(cbind(K,L), 1, paste, collapse= ":")
M[M == "a:a"] = sample(LV3, length(which(M == "a:a")), prob = c(0.33, 0.33, 0.34), replace = TRUE)
M[M == "a:b"] = sample(LV3, length(which(M == "a:b")), prob = c(0.2, 0.2, 0.6), replace = TRUE)
M[M == "a:c"] = sample(LV3, length(which(M == "a:c")), prob = c(0.1, 0.75, 0.15), replace = TRUE)
M[M == "b:a"] = sample(LV3, length(which(M == "b:a")), prob = c(0.7, 0.18, 0.12), replace = TRUE)
M[M == "b:b"] = sample(LV3, length(which(M == "b:b")), prob = c(0.9, 0.05, 0.05), replace = TRUE)
M[M == "b:c"] = sample(LV3, length(which(M == "b:c")), prob = c(0.8, 0.14, 0.06), replace = TRUE)
M[M == "c:a"] = sample(LV3, length(which(M == "c:a")), prob = c(0.15, 0.15, 0.7), replace = TRUE)
M[M == "c:b"] = sample(LV3, length(which(M == "c:b")), prob = c(0.5, 0.25, 0.25), replace = TRUE)
M[M == "c:c"] = sample(LV3, length(which(M == "c:c")), prob = c(0.15, 0.45, 0.4), replace = TRUE)


N = apply(cbind(M,J), 1, paste, collapse= ":")
N[N == "a:a"] = sample(LV3, length(which(N == "a:a")), prob = c(0.8, 0.1, 0.1), replace = TRUE)
N[N == "a:b"] = sample(LV3, length(which(N == "a:b")), prob = c(0.2, 0.1, 0.7), replace = TRUE)
N[N == "a:c"] = sample(LV3, length(which(N == "a:c")), prob = c(0.05, 0.05, 0.9), replace = TRUE)
N[N == "b:a"] = sample(LV3, length(which(N == "b:a")), prob = c(0.9, 0.08, 0.02), replace = TRUE)
N[N == "b:b"] = sample(LV3, length(which(N == "b:b")), prob = c(0.6, 0.05, 0.35), replace = TRUE)
N[N == "b:c"] = sample(LV3, length(which(N == "b:c")), prob = c(0.13, 0.17, 0.7), replace = TRUE)
N[N == "c:a"] = sample(LV3, length(which(N == "c:a")), prob = c(0.75, 0.1, 0.15), replace = TRUE)
N[N == "c:b"] = sample(LV3, length(which(N == "c:b")), prob = c(0.05, 0.5, 0.45), replace = TRUE)
N[N == "c:c"] = sample(LV3, length(which(N == "c:c")), prob = c(0.1, 0.35, 0.55), replace = TRUE)

X = apply(cbind(J, C2), 1, paste, collapse= ":")
X[X == "a:a"] = sample(LV3, length(which(X == "a:a")), prob = c(0.13, 0.17, 0.7), replace = TRUE)
X[X == "a:b"] = sample(LV3, length(which(X == "a:b")), prob = c(0.9, 0.05, 0.05), replace = TRUE)
X[X == "b:a"] = sample(LV3, length(which(X == "b:a")), prob = c(0.2, 0.2, 0.6), replace = TRUE)
X[X == "b:b"] = sample(LV3, length(which(X == "b:b")), prob = c(0.13, 0.17, 0.7), replace = TRUE)
X[X == "c:a"] = sample(LV3, length(which(X == "c:a")), prob = c(0.1, 0.35, 0.55), replace = TRUE)
X[X == "c:b"] = sample(LV3, length(which(X == "c:b")), prob = c(0.25, 0.5, 0.25), replace = TRUE)

T = X
T[T == "a"] = sample(LV3, length(which(T == "a")), prob = c(0.7, 0.15, 0.15), replace = TRUE)
T[T == "b"] = sample(LV3, length(which(T == "b")), prob = c(0.25, 0.5, 0.25), replace = TRUE)
T[T == "c"] = sample(LV3, length(which(T == "c")), prob = c(0.45, 0.05, 0.5), replace = TRUE)

Y = apply(cbind(T,C1), 1, paste, collapse= ":")
Y[Y == "a:a"] = sample(LV3, length(which(Y == "a:a")), prob = c(0.8, 0.1, 0.1), replace = TRUE)
Y[Y == "a:b"] = sample(LV3, length(which(Y == "a:b")), prob = c(0.2, 0.1, 0.7), replace = TRUE)
Y[Y == "b:a"] = sample(LV3, length(which(Y == "b:a")), prob = c(0.9, 0.08, 0.02), replace = TRUE)
Y[Y == "b:b"] = sample(LV3, length(which(Y == "b:b")), prob = c(0.6, 0.05, 0.35), replace = TRUE)
Y[Y == "c:a"] = sample(LV3, length(which(Y == "c:a")), prob = c(0.75, 0.1, 0.15), replace = TRUE)
Y[Y == "c:b"] = sample(LV3, length(which(Y == "c:b")), prob = c(0.05, 0.5, 0.45), replace = TRUE)


P = T
P[P == "a"] = sample(LV3, length(which(P == "a")), prob = c(0.7, 0.15, 0.15), replace = TRUE)
P[P == "b"] = sample(LV3, length(which(P == "b")), prob = c(0.25, 0.5, 0.25), replace = TRUE)
P[P == "c"] = sample(LV3, length(which(P == "c")), prob = c(0.45, 0.05, 0.5), replace = TRUE)


Q = P
Q[Q == "a"] = sample(LV3, length(which(Q == "a")), prob = c(0.7, 0.15, 0.15), replace = TRUE)
Q[Q == "b"] = sample(LV3, length(which(Q == "b")), prob = c(0.25, 0.5, 0.25), replace = TRUE)
Q[Q == "c"] = sample(LV3, length(which(Q == "c")), prob = c(0.45, 0.05, 0.5), replace = TRUE)

learning.test = data.frame(
  A = factor(a, levels = LV3),
  B = factor(b, levels = LV3),
  C = factor(c, levels = LV3),
  D = factor(d, levels = LV3),
  E =  factor(e, levels = LV3),
  F = factor(f, levels = c("a", "b"))
)

# network specification.
dag = model2network("[U][C1|U][C2|U][F][G][H][D|F][E|G:H][I|E][B|C2:D][K][J|K][L|K][M|K:L][N|J:M][X|C2:J][T|X][Y|C1:T][P|T][Q|P]")


bn = custom.fit(dag, list(
  F = matrix(rep(1/3, 3), ncol = 3, dimnames = list(NULL, LV3)),
  G = matrix(c(0.75, 0.2, 0.05), ncol = 3, dimnames = list(NULL, LV3)),
  H = matrix(c(0.6, 0.2, 0.2), ncol = 3, dimnames = list(NULL, LV3)),
  K = matrix(c(0.15, 0.25, 0.6), ncol = 3, dimnames = list(NULL, LV3)),
  U = matrix(rep(1/2, 2), ncol = 2, dimnames = list(NULL, c("a", "b"))),
              
  C1 = array(c(0.8, 0.1, 0.1, 0.4, 0.2, 0.4), dim = c(2,2),
             dimnames = list(C1 = c("a", "b"), U = c("a", "b"))),
  C2 = array(c(0.8, 0.1, 0.1, 0.4, 0.2, 0.4), dim = c(2,2),
             dimnames = list(C2 = c("a", "b"), U = c("a", "b"))),
  D = matrix(c(0.4, 0.3, 0.3, 0.2, 0.4, 0.4, 0.1, 0.8, 0.1), ncol = 3,
             dimnames = list(D = LV3, F = LV3)),
  L = matrix(c(0.2, 0.7, 0.1, 0.2, 0.1, 0.7, 0.4, 0.3, 0.2), ncol = 3,
             dimnames = list(L = LV3, K = LV3)),
  J = matrix(c(0.2, 0.4, 0.4, 0.3, 0.4, 0.2, 0.1, 0.8, 0.1), ncol = 3,
             dimnames = list(J = LV3, K = LV3)),
  
  E = array(c(0.8, 0.1, 0.1, 0.2, 0.1, 0.7, 0.4, 0.2, 0.4, 0.1,
              0.8, 0.1, 0.9, 0.05, 0.05,0.3, 0.4, 0.3, 0.1, 0.1, 0.8, 0.25, 0.5, 0.25, 0.15, 0.45, 0.4), dim = c(3, 3, 3), dimnames = list(E = LV3, G = LV3, H = LV3)),
  B = array(c(0.8, 0.1, 0.1, 0.4, 0.5, 0.1, 0.2, 0.2, 0.6, 0.3, 0.4, 0.3,
              0.1, 0.1, 0.8, 0.25, 0.5, 0.25), dim = c(3, 3, 2), dimnames = list(B = LV3, D = LV3, C2 = c("a", "b"))),
  I = matrix(c(0.3, 0.4, 0.3,0.2, 0.4, 0.4, 0.1, 0.8, 0.1), ncol = 3,
             dimnames = list(I = LV3, E = LV3)),
  M = array(c(0.33, 0.33, 0.34, 0.2, 0.2, 0.6, 0.1, 0.75, 0.15, 0.7, 0.18, 0.12,
              0.9, 0.05, 0.05, 0.8, 0.14, 0.06, 0.15, 0.15, 0.7, 0.5, 0.25, 0.25, 0.15, 0.45, 0.4), dim = c(3, 3, 3), dimnames = list(M = LV3, K = LV3, L = LV3)),
  N = array(c(0.8, 0.1, 0.1, 0.2, 0.1, 0.7,0.05, 0.05, 0.9, 0.9, 0.08, 0.02, 0.6, 0.05, 0.35,
              0.13, 0.17, 0.7, 0.75, 0.1, 0.15, 0.05, 0.5, 0.45, 0.1, 0.35, 0.55), dim = c(3, 3, 3), dimnames = list(N = LV3, M = LV3, J = LV3)),
  
  X = array(c(0.13, 0.17, 0.7, 0.9, 0.05, 0.05, 0.2, 0.2, 0.6, 0.13, 0.17, 0.7,
              0.1, 0.35, 0.55, 0.25, 0.5, 0.25), dim = c(3, 3, 2), dimnames = list(X = LV3, J = LV3, C2 = c("a", "b"))),
  T = matrix(c(0.7, 0.15, 0.15, 0.25, 0.5, 0.25, 0.45, 0.05, 0.5), ncol = 3,
             dimnames = list(T = LV3, X = LV3)),
  Y = array(c(0.8, 0.1, 0.1, 0.2, 0.1, 0.7, 0.9, 0.08, 0.02, 0.6, 0.05, 0.35,
              0.75, 0.1, 0.15, 0.05, 0.5, 0.45), dim = c(3, 3, 2), dimnames = list(Y = LV3, T = LV3, C1 = c("a", "b"))),
  
  P = matrix(c(0.7, 0.15, 0.15, 0.25, 0.5, 0.25, 0.45, 0.05, 0.5), ncol = 3,
             dimnames = list(P = LV3, T = LV3)),
  Q = matrix(c(0.7, 0.15, 0.15, 0.25, 0.5, 0.25, 0.45, 0.05, 0.5), ncol = 3,
             dimnames = list(Q = LV3, P = LV3))))

df<- rbn(bn,1000)

C1 = array(c(0.8, 0.1, 0.1, 0.4, 0.2, 0.4), dim = c(2,2),
            dimnames = list(C1 = c("a", "b"), U = c("a", "b")))