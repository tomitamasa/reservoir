#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "random.h"

#define N 10
#define T 200
#define RHO 0.8

/////////////////////////////
////// 表示関係
////////////////////////////
// 行列をいい感じにコンソールに出力する
void print_matrix(int m, int n, double mat_mn[m][n])
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%3.2lf\t", mat_mn[i][j]);
        }
        printf("\n");
    }
}

// ベクトルをいい感じにコンソールに表示する
void print_vector(int n, double vec[n])
{
    for (int i = 0; i < n; i++)
    {
        printf("%5.3lf\t", vec[i]);
    }
    printf("\n");
}

// CSV(カンマ区切り)ベクトルを表示する
void print_vector_csv(int n, double vec[n])
{
    for (int i = 0; i < n; i++)
    {
        printf("%lf", vec[i]);
        // カンマの出力制御(もっといい感じによろしく)
        if (i != n - 1)
        {
            printf(",");
        }
    }
    printf("\n");
}

// CSV(カンマ区切り)ベクトルでファイルポインタに書き込む
void fprint_vector_csv(int n, double vec[n], FILE *fp)
{
    for (int i = 0; i < n; i++)
    {
        fprintf(fp, "%lf", vec[i]);
        // カンマの出力制御(もっといい感じによろしく)
        if (i != n - 1)
        {
            fprintf(fp, ",");
        }
    }
    fprintf(fp, "\n");
}

void print_y_y_train(int bat_num, int len, double y[], double y_expect[], int verbose)
{
    printf("---------------------------------\n");
    printf("-------------BAT [%2d]------------\n", bat_num);
    if (verbose == 1)
    {
        printf("y(t)=\n");
        print_vector(T, y);
        printf("y_expect(t)\n");
        print_vector(T, y_expect);
    }
    else
    {
        print_vector_csv(T, y);
        print_vector_csv(T, y_expect);
    }
    printf("---------------------------------\n");
}

/////////////////////////////
////// ベクトル，行列の計算関係
////////////////////////////
// ベクトルの内積（inner product）
double product(int len, double x[len], double y[len])
{
    double res = 0;
    for (int i = 0; i < len; i++)
    {
        res += x[i] * y[i];
    }
    return res;
}

// 行列を縦にスライスする (pythonのx[ : , idx]みたいな感じ)
void mat_vvec_slice(int m, int n, double mat_mn[m][n], int idx, double dst[m])
{
    // 2次元の行列から　指定のインデックスidx　の列ベクトルをdstに代入する．
    for (int i = 0; i < m; i++)
    {
        dst[i] = mat_mn[i][idx];
    }
}

// (m, n)行列とnベクトルの積
void mul_mat_vec(int m, int n, double mat_mn[m][n], double vec[n], double dst[m])
{
    // 出力はm次元ベクトル
    for (int i = 0; i < m; i++)
    {
        dst[i] = product(n, mat_mn[i], vec);
    }
}

// ベクトルのスカラー倍
void mul_vec_scaler(int n, double vec[n], double scaler, double dst[n])
{
    // 処理先dstと処理元vecは同じでもいい
    for (int i = 0; i < n; i++)
    {
        dst[i] = scaler * vec[i];
    }
}

// (m, n)行列と(n, l)行列の積
void mul_mat_mat(int m, int n, int l, double mat_mn[m][n], double mat_nl[n][l], double dst[m][l])
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < l; j++)
        {
            double vec[n];
            mat_vvec_slice(n, l, mat_nl, j, vec);
            dst[i][j] = product(n, mat_mn[i], vec);
        }
    }
}

/////////////////////////////
//////  最小二乗法のため
////// (ここでは正規方程式の掃き出し法で行う）
////// ここの関数は基本的に副作用（演算元の行列を直接書き換える）がある！
////////////////////////////
// ピボット操作
void pivot(int m, int n, double mat_mn[m][n], int row, int col)
{
    // 対象とする列(col)において，最も大きい成分を持つ行を指定した行(row)と入れ替える処理を行う
    double th = fabs(mat_mn[row][col]);
    int swap_idx = row;

    // 指定の列で最も大きい絶対値を持つ行を選択する
    for (int i = row + 1; i < m; i++)
    {
        if (fabs(mat_mn[i][col]) > th)
        {
            th = fabs(mat_mn[i][col]);
            swap_idx = i;
        }
    }

    // 上で求めた行と引数で指定した行が同じなら入れ替えしない
    if (swap_idx == col)
    {
        return;
    }

    // 上で求めた行と指定の行を入れ替えを要素ごとに行う
    double tmp;
    for (int i = 0; i < n; i++)
    {
        tmp = mat_mn[col][i];
        mat_mn[col][i] = mat_mn[swap_idx][i];
        mat_mn[swap_idx][i] = tmp;
    }
}

//  行列の(row , col)成分が1となるように，row行目をスカラ倍する
void normalize_mat_row(int m, int n, double mat_mn[m][n], int row, int col)
{
    double scale = 1.0 / mat_mn[row][col];
    for (int i = 0; i < n; i++)
    {
        mat_mn[row][i] *= scale;
    }
}

// (row, col)成分を基準とした掃き出し処理
void row_reduction(int m, int n, double mat_mn[m][n], int row, int col)
{
    for (int i = 0; i < m; i++)
    {
        if (i == row)
        {
            continue;
        }
        double scale = mat_mn[i][col] / mat_mn[row][col];
        for (int j = 0; j < n; j++)
        {
            mat_mn[i][j] -= scale * mat_mn[row][j];
        }
    }
}

// 最小二乗法
void LeastSquareMethod(int m, int n, double A[m][n], double b[m], double res[n])
{
    // 以下，Ax = y という一般形をもとに変数名がついている

    // 係数行列Aの転置行列を計算
    double At[n][m];
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            At[i][j] = A[j][i];
        }
    }

    // 拡大係数行列を計算 (A | y )
    double Ay[m][n + 1];
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n + 1; j++)
        {
            if (j == n)
            {
                Ay[i][j] = b[i];
            }
            else
            {
                Ay[i][j] = A[i][j];
            }
        }
    }

    // 正規方程式 A^t Ay
    double At_Ay[n][n + 1];
    mul_mat_mat(n, m, n + 1, At, Ay, At_Ay);

    // 掃き出し法
    for (int i = 0; i < n; i++)
    {
        pivot(n, n + 1, At_Ay, i, i);

        // ピポット後に対角成分が0近くになっていたらランク落ちしてると思う（解釈違うかも）
        // とりあえず，この行は飛ばして掃き出し法は続行する
        if (fabs(At_Ay[i][i]) < 1e-10)
        {
            printf("value error!!!!!!!!!!!!!!!!!!!!!!\n At_Ay[%d][%d] = %lf \n", i, i, At_Ay[i][i]);
            print_matrix(n, n + 1, At_Ay);
            continue;
        }
        // 対角成分(i, i)を1とするように，i行目をスカラ倍する
        normalize_mat_row(n, n + 1, At_Ay, i, i);
        // (i, i)成分を基準に掃き出し処理をする
        row_reduction(n, n + 1, At_Ay, i, i);
    }

    // 出力ベクトル(res)にまとめる
    mat_vvec_slice(n, n + 1, At_Ay, n, res);
}

/////////////////////////////
//////  教師データ生成用の適当な写像
////////////////////////////
// ロジスティック写像
double logistic_function(double x0, double a)
{
    return a * x0 * (1 - x0);
}

// エノン写像（複数戻り値がよくわからんかったので，ポインタで直接書き換える）
// いい感じに配列とかで返せそうならそれでも良いのでは？
void normal_henon_map(double *x0, double *y0)
{
    double a = 1.4;
    double b = 0.3;
    double tmp = *x0;
    *x0 = 1 - a * (*x0) * (*x0) + (*y0);
    *y0 = b * tmp;
}

/////////////////////////////
//////  結果の処理
////////////////////////////
// 平均二乗誤差（リザバーの出力と教師データの二乗誤差平均を返す）
double MSE(int len, double y[], double y_train[])
{
    double mse = 0;

    for (int i = 0; i < len; i++)
    {
        double diff = y[i] - y_train[i];
        mse += diff * diff;
    }

    return mse / len;
}

void reservoir(double W_in[N], double W_out[N], double W_r[N][N], double x[N], double u[T], double X[T][N], double y[T])
{
    //////////////
    /// 以下繰り返し処理を書く
    for (int t = 0; t < T; t++)
    {
        // <U(t)の計算>
        double U1[N], U2[N];
        mul_vec_scaler(N, W_in, u[t], U1);
        mul_mat_vec(N, N, W_r, x, U2);
        // </ U(t)の計算>

        // <x(t)の計算>
        for (int i = 0; i < N; i++)
        {
            x[i] = (1 - RHO) * x[i] + RHO * tanh(U1[i] + U2[i]);
        }
        // </x(t)の計算>

        ///
        // 後で使うために値を保存する
        // ニューロンの状態X(t)
        for (int i = 0; i < N; i++)
        {
            X[t][i] = x[i];
        }
        // リザバーの出力y(t)
        y[t] = product(N, W_out, x);
    }
}

// 再帰的予測
void reservoir_recursive_prediction(double W_in[N], double W_out[N], double W_r[N][N], double x[N], double u[T], double X[T][N], double y[T], int prediction_start_idx)
{
    // u[0]にだけ値があればよい
    //////////////
    /// 以下繰り返し処理を書く
    for (int t = 0; t < T; t++)
    {
        double input_data;
        // 更新回数により，教師データ or 再帰的な入力を切り替える
        // 最初は教師データ
        if (t < prediction_start_idx)
        {
            input_data = u[t];
        }
        else
        {
            // 一時刻前の出力
            input_data = y[t - 1];
        }
        // <U(t)の計算>
        double U1[N], U2[N];
        mul_vec_scaler(N, W_in, input_data, U1);
        mul_mat_vec(N, N, W_r, x, U2);
        // </ U(t)の計算>

        // <x(t)の計算>
        for (int i = 0; i < N; i++)
        {
            x[i] = (1 - RHO) * x[i] + RHO * tanh(U1[i] + U2[i]);
        }
        // </x(t)の計算>

        for (int i = 0; i < N; i++)
        {
            X[t][i] = x[i];
        }
        y[t] = product(N, W_out, x);
    }
}

/////////////////////////////
//////  メインの処理
////////////////////////////
int main(void)
{
    int batch_num = 10;

    double W_in[N];
    double W_out[N];
    double W_r[N][N];

    double x[N];
    double X[T][N];
    double y[T];
    double y_train[T];
    double u[T];

    // 乱数のシード設定
    // srand((unsigned int)time(NULL));
    init_genrand((unsigned)time(NULL));

    // <入力重みW_inの初期化>
    for (int i = 0; i < N; i++)
    {
        double th = genrand_real1();
        if (th < 0.25)
        {
            W_in[i] = -0.5;
        }
        else if (th > 0.75)
        {
            W_in[i] = 0.5;
        }
        else
        {
            W_in[i] = 0;
        }
    }
    printf("W_in = \n");
    print_vector(N, W_in);
    // </W_inの初期化>

    // <出力重みW_outの初期化>
    // これでいいかわからん
    for (int i = 0; i < N; i++)
    {
        W_out[i] = rand_normal(0, 1);
    }
    printf("W_out = \n");
    print_vector(N, W_out);
    // </W_outの初期化>

    // <ニューロン間の結合W_rの初期化>
    for (int i = 0; i < N; i++)
    {
        for (int j = i; j < N; j++)
        {
            if (genrand_real1() < 0.3)
            {
                W_r[i][j] = W_r[j][i] = rand_normal(0, 1);
            }
            else
            {
                W_r[i][j] = W_r[j][i] = 0;
            }
        }
    }
    printf("W_r = \n");
    print_matrix(N, N, W_r);
    // </W_rの初期化>

    // <ニューロンの状態x(0)の初期化>
    // 初期の内部状態は0でいいよね？（雑）
    for (int i = 0; i < N; i++)
    {
        x[i] = rand_normal(0, 1);
    }
    // </xの初期化>

    //////////
    // バッチ学習
    //////////
    // エノン写像の初期値
    double x0 = 0.1;
    double y0 = 0;
    double mse[batch_num];
    for (int bat = 0; bat < batch_num; bat++)
    {
        printf("------------------batch[%d]-------------------\n", bat);
        // <入力u(t), 教師データy_train()の生成>
        double *xy;
        for (int t = 0; t < T; t++)
        {
            // ロジスティックマップの１ステップ予測を行う
            // u[t] = x0;
            // x0 = logistic_function(x0, 3.999);
            // y_train[t] = x0;

            // 入力データはエノン写像のx[t]
            u[t] = x0;
            normal_henon_map(&x0, &y0);
            // 教師データはエノン写像のx[t+1]
            y_train[t] = x0;
        }
        // </u(t)の生成>

        reservoir(W_in, W_out, W_r, x, u, X, y);

        // 最小二乗法（バッチ処理）よりWoutを最適化する
        LeastSquareMethod(T, N, X, y_train, W_out);

        // 最適化した重み
        printf("after Wout=\n");
        print_vector(N, W_out);

        // コンソールにいい感じに出力するだけ
        // print_y_y_train(bat, T, y, y_train, 0);

        // MSEを記録する
        mse[bat] = MSE(T, y, y_train);
    }
    printf("MSE_trajectory=\n");
    print_vector(batch_num, mse);

    //////////
    // 再帰的な1ステップ予測
    //////////
    // リザバの内部状態を0 で初期化した
    mul_vec_scaler(N, x, 0, x);

    int recursive_prediction_start_idx = 200;
    // 超適当なエノン写像による正解データ生成の初期値
    double y_ans[T];
    x0 = 0.01;
    y0 = 0.02;
    // 再帰的予測に使うためのデータを生成する
    for (int t = 0; t < T; t++)
    {
        // 入力データはエノン写像のx[t]
        if (t < recursive_prediction_start_idx)
        {
            u[t] = x0;
        }
        else
        {
            u[t] = 0; // 別に不定値で良いが，精神衛生上0にしておく
        }
        normal_henon_map(&x0, &y0);

        // 正解データはエノン写像のx[t+1].一応保存しておく
        y_ans[t] = x0;
    }

    printf("----------start prediction------------\n");
    reservoir_recursive_prediction(W_in, W_out, W_r, x, u, X, y, recursive_prediction_start_idx);

    printf("---------------------------------\n");
    int verbose = 2; //  1: Console Display style ,  2: CSV style,   other : None
    if (verbose > 0)
    {
        printf("y(t)=\n");
        print_vector(T, y);
        printf("y_expect(t)\n");
        print_vector(T, y_ans);
    }
    if (verbose > 1)
    {
        // 2以上ならCSVにも書き込む
        FILE *fp;
        fp = fopen("result_recursive.csv", "w");
        fprint_vector_csv(T, u, fp);
        fprint_vector_csv(T, y, fp);
        fprint_vector_csv(T, y_ans, fp);
        fclose(fp);
    }
    printf("---------------------------------\n");
}
