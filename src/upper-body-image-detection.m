classdef question3
    methods (Static)
        function question_3_4_2(trLbs, trD)
            cmd = ['-v 5', '-t 2'];
            cv = svmtrain(trLbs, trD, cmd);
            fprintf('Cross validation Accuracy for default values of C and gamma : %g\n', cv);
        end

        function question_3_4_3(trLbs, trD)
            bestcv = 0;
            bestc = 0;
            bestg = 0;
            for c = 8:15
               for g = -4:4
                 params = ['-v 5 -c ', num2str(2^c), ' -g ', num2str(2^g), '-t 2'];
                 cv = svmtrain(trLbs, trD, params);
                 if (cv >= bestcv)
                   bestcv = cv; bestc = 2^c; bestg = 2^g;
                 end
                 fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', c, g, cv, bestc, bestg, bestcv);
               end
            end
            fprintf('Cross validation Accuracy for best values of C : %g and gamma : %g is %g\n',bestc, bestg, bestcv');  
        end

        function [trainK, testK, bestc] = question_3_4_5(trD, tstD, trLbs)
            bestc = 0; bestcv = 0; bestg = 0;
            for gamma = -4:5
                [trainK, testK] = question3.cmpExpX2Kernel(trD, tstD, 2^gamma);

                for c = 6:15
                    cmd = ['-v 5 -c ', num2str(2^c), '-t 4'];
                    cv = svmtrain(trLbs, trainK, cmd);
                    if (cv >= bestcv)
                      bestcv = cv; bestc = 2^c; bestg = 2^gamma;
                    end
                    fprintf('%g  %g  %g(best c=%g, rate=%g, gamma=%g)\n', c, cv, gamma,  bestc,bestcv, bestg);
                end
            end
            fprintf('Best C : %d,  gamma : %s and 5 - fold Cross Validation accuracy : %s for Exponential Chi-square kernel ', bestc, bestg, bestcv);
        end


        function question_3_4_6(trainK, testK, trLbs, tstD , bestc)
            params = ['-c ', num2str(bestc), '-t 4'];
            model = svmtrain(trLbs, trainK, params);
            [predTrainLabel, acc, decVals] = svmpredict(trLbs, trainK, model);
            fprintf('Training accuracy: %g\n', acc);
            tstLb = zeros(size(tstD, 1),1);
            [predTestLabel] = svmpredict(tstLb, testK, model);
            mat_index = reshape( 1:1600, 1600, 1);
            final_mat = [mat_index, predTestLabel];
            disp(size(final_mat));
            T = array2table(final_mat,'VariableNames',{'ImgId','Prediction'});
            writetable(T,'output.csv');
        end

        function [trainK, testK] = cmpExpX2Kernel(trD, tstD, gamma)
            numTrain = size(trD,1); numTest = size(tstD,1);
            trainK =  [ (numTrain:1) , question3.chiSquareKernel(trD,trD, gamma) ];
            testK = [ (numTest:1)  , question3.chiSquareKernel(tstD,trD, gamma) ];
        end


        function kernel = chiSquareKernel(train1, train2, gamma)
            kernel = zeros(size(train1,1), size(train2,1));
            for i = 1:size(train1, 1)
                for j = 1:size(train2, 1)
                    x = train1(i,:);
                    y = train2(j, :);
                    a=(x-y).^2;
                    b=(x+y);
                    summation = sum(a./(b + eps));
                    kernel(i,j) = summation;
                end
            end
            kernel = exp((-1/gamma) * kernel);
        end
    end
    
end











