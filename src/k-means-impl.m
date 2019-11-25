X = load('../digit/digit.txt');
Y = load('../digit/labels.txt');
iterations = 20;

%question_2_5_1(X, Y, iterations);
%question_2_5_2(X, iterations);
question_2_5_3(X, iterations);
% question_2_5_4(X, Y, iterations);

function question_2_5_1(X, Y, iterations)
    K = [2,4,6];

    for k_val = 1:size(K,2)
        prev_cluster_centre = [];

        for i=1:K(k_val)
            prev_cluster_centre = [prev_cluster_centre;X(i,:)];
        end

        [new_cluster_classify, new_cluster_centre, iter] = k_means.k_means_impl(iterations, X, K(k_val), prev_cluster_centre);

        sum_of_squares = calc_ss(new_cluster_classify, new_cluster_centre, X, K(k_val));
        fprintf("sum of squares for k : %d is %d\n", K(k_val), sum_of_squares);

        [p1, p2, p3] = pair_counting(Y, new_cluster_classify);
        fprintf("pair counting for k : %d is p1 = %d, p2 = %d, p3 = %d\n\n", K(k_val), p1, p2, p3);
    end
end

function question_2_5_2(X, iterations)
    K = 6;

    prev_cluster_centre = [];

    for i=1:K
        prev_cluster_centre = [prev_cluster_centre;X(i,:)];
    end

    [new_cluster_classify, new_cluster_centre, iter] = k_means.k_means_impl(iterations, X, K, prev_cluster_centre);
    fprintf("No. of iterations for K = %d is %d\n", K, iter);

end


function question_2_5_3(X, iterations)
    k_sum_of_squares = [];
    K = [1,2,3,4,5,6,7,8,9,10];

    for k_val = 1:size(K,2)
        sum_of_squares = 0;
        for each_iter = 1:10
            prev_cluster_centre = [];
            prev_cluster_centre = X(randperm(size(X, 1), K(k_val)), :);
            [new_cluster_classify, new_cluster_centre, ~] = k_means.k_means_impl(iterations, X, K(k_val), prev_cluster_centre); 
            sum_of_squares = sum_of_squares + calc_ss(new_cluster_classify, new_cluster_centre, X, K(k_val));
            
        end
        
        k_sum_of_squares = [k_sum_of_squares, sum_of_squares/10];
    end
    plot(k_sum_of_squares, '-o');
    title('Sum of squares Vs K')
    xlabel('K')
    ylabel('sum of squares')
end

function question_2_5_4(X, Y, iterations)
    all_p1 = [];
    all_p2 = [];
    all_p3 = [];
    K = [1,2,3,4,5,6,7,8,9,10];

    for k_val = 1:size(K,2)
        total_p1 = 0;
        total_p2 = 0;
        total_p3 = 0;
        for each_iter = 1:10
            prev_cluster_centre = [];
            prev_cluster_centre = X(randperm(size(X, 1), K(k_val)), :);
            [new_cluster_classify, new_cluster_centre, ~] = k_means.k_means_impl(iterations, X, K(k_val), prev_cluster_centre); 
            [p1, p2, p3] = pair_counting(Y, new_cluster_classify);
            total_p1 = total_p1 + p1;
            total_p2 = total_p2 + p2;
            total_p3 = total_p3 + p3;
        end
        
        all_p1 = [all_p1, total_p1/10];
        all_p2 = [all_p2, total_p2/10];
        all_p3 = [all_p3, total_p3/10];
    end
    plot(K, all_p1, K, all_p2, K, all_p3);
    title('Pair Counting Vs K');
    xlabel('K');
    ylabel('Pair Counts');
    legend({'p1','p2','p3'});
end

function sum_of_squares = calc_ss(new_cluster_classify, new_cluster_c, X, K)
    sum_of_squares = 0;
    for cluster = 1:K
        cluster_X = X(find(new_cluster_classify == cluster),:);
        for each_x = 1:size(cluster_X, 1)
            sum_of_squares = sum_of_squares + ((k_means.euclidean_distance(cluster_X(each_x,:), new_cluster_c(cluster,:))).^2);
        end
    end   
end



function [p1, p2, p3] = pair_counting(actual_cluster, pred_cluster)
    p1 = 0;
    p2 = 0;
    total_p1 = 0;
    total_p2 = 0;

    for i = 1:size(actual_cluster, 1)
        for j = i + 1:size(actual_cluster, 1)
            if actual_cluster(i) == actual_cluster(j)
                total_p1 = total_p1 + 1;
            else
                total_p2 = total_p2 + 1;
            end

            if actual_cluster(i) == actual_cluster(j) && pred_cluster(i) == pred_cluster(j)
                p1 = p1 + 1;
            end
            if actual_cluster(i) ~= actual_cluster(j) && pred_cluster(i) ~= pred_cluster(j)
                p2 = p2 + 1;
            end
        end
    end
    p1 = p1 * 100 / total_p1;
    p2 = p2 * 100 /total_p2;
    p3 = (p1 + p2)/2;
end


    
    