%% określenie rozmiarów macierzy
m = 37 * 64;
n = 97 * 64;
%% utworzenie macierzy i iloczynu
A = randn(m, n);
C = A * A.';
%% zapamiętanie macierzy w pliku binarnym
f = fopen('matmultran.dat', 'wb');
fwrite(f, [m, n], 'int');
% zapamiętujemy transponowane macierze, bo język C
% przechowuje macierze wierszami, a nie kolumnami
fwrite(f, A.', 'single');
fwrite(f, C.', 'single');
fclose(f);
