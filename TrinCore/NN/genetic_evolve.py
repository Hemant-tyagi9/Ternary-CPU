import numpy as np
from pathlib import Path
from .nn_loader import NeuroGate
import os

class everything_under:
    def score_model(weights, biases, op):
        # evaluate on exhaustive dataset
        from src.train.train_gate import prepare_dataset
        X,Y = prepare_dataset(op)
        # forward
        h = np.tanh(X.dot(weights[0].T) + biases[0])
        logits = np.tanh(h.dot(weights[1].T) + biases[1])
        preds = logits.argmax(axis=1)
        targets = Y.argmax(axis=1)
        return (preds==targets).mean()

    def random_population(pop, input_dim=6, hidden=16):
        pop_list = []
        for _ in range(pop):
            W1 = np.random.randn(hidden, input_dim).astype(np.float32)*0.2
            b1 = np.zeros((hidden,), dtype=np.float32)
            W2 = np.random.randn(3,hidden).astype(np.float32)*0.2
            b2 = np.zeros((3,), dtype=np.float32)
            pop_list.append( ( [W1,W2], [b1,b2] ) )
        return pop_list

    def mutate(weights, biases, scale=0.1):
        newW = [w + np.random.randn(*w.shape).astype(np.float32)*scale for w in weights]
        newb = [b + np.random.randn(*b.shape).astype(np.float32)*scale for b in biases]
        return newW, newb

    def evolve(op='AND', generations=50, pop=30):
        population = random_population(pop)
        best = None
        for g in range(generations):
            scored = []
            for indiv in population:
                sc = score_model(indiv[0], indiv[1], op)
                scored.append( (sc, indiv) )
            scored.sort(key=lambda x: x[0], reverse=True)
            best_score, best_indiv = scored[0]
            print(f"Gen {g} best {best_score:.3f}")
            if best_score==1.0:
                best = best_indiv
                break
            # selection top 10
            top = [ind for _,ind in scored[:10]]
            # produce new pop by mutating top
            population = top.copy()
            while len(population) < pop:
                parent = top[np.random.randint(0, len(top))]
                child = mutate(parent[0], parent[1], scale=0.05)
                population.append(child)
            best = best_indiv
        # save best
        p = Path("models"); p.mkdir(exist_ok=True)
        W0,W1 = best[0]
        b0,b1 = best[1]
        np.savez_compressed(p / f"{op.lower()}_ga_best.npz", n_layers=2, W0=W0, b0=b0, W1=W1, b1=b1)
        print("Saved GA best model for", op)
        
if __name__ == '__main__':
    evolve('AND', generations=30, pop=40)
