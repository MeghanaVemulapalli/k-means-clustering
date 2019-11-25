classdef k_means
    methods (Static)
        function [new_cluster_classify, new_cluster_centre, iter] = k_means_impl(iterations, X, K, prev_cluster_centre)
            prev_cluster_classify = zeros(size(X,1),1);
            for iter= 1:iterations
                
                isOptimal = true;
                new_cluster_classify = [];
                new_cluster_centre = [];
                for eachpoint = 1:size(X,1)
                    dist = [];
                    for cluster = 1:K
                        dist = [dist;k_means.euclidean_distance(X(eachpoint,:), prev_cluster_centre(cluster,:))];
                    end
                    min_vals = find(dist == min(dist));
                    new_cluster_classify = [new_cluster_classify;min_vals(1)];
                end

                new_cluster_centre = k_means.calc_new_clust_centre(new_cluster_classify, K, X);

                for i = 1:size(new_cluster_classify,1)
                    if new_cluster_classify(i) ~= prev_cluster_classify(i)
                        isOptimal = false;
                        break;
                    end
                end

                if isOptimal
                    break;
                end

                prev_cluster_centre = new_cluster_centre;
                prev_cluster_classify = new_cluster_classify;

            end
        end
        
        function euclid_dist = euclidean_distance(x, y)
            distance = 0;
            for i=1:size(x,2)
                distance = distance + (x(i) - y(i)).^2;
            end
            euclid_dist = sqrt(distance);
        end
        
        function new_cluster_c = calc_new_clust_centre(new_cluster_classify, K, X)
            for cluster = 1:K
                    cluster_X = X(find(new_cluster_classify == cluster),:);
                    new_cluster_c(cluster,:) = mean(cluster_X);
            end
        end
        
    end
end