** Quiz for "Absolute Trick" **
Let's say that we have a line whose equation is y = -0.6x + 4. For the point (x,y) = (-5, 3), apply the absolute trick to get the new equation for the line, using a learning rate of alpha = 0.1alpha=0.1.
Report your answer in the form y = w_1x + w_2, substituting appropriate values for w_1 and w_2.

y=-.1x+3.9


** Quiz for "Square Trick" **
Let's say that we have a line whose equation is y = -0.6x + 4. For the point (x,y) = (-5, 3), apply the square trick to get the new equation for the line, using a learning rate of alpha = 0.01alpha=0.01.
Report your answer in the form y = w_1x + w_2, substituting appropriate values for w_1 and w_2.

q' = w1*p+4 = -0.6*(-5) + 4 = 3+4 = 7 y = ( w1 + p*(q-q')*alpha )x + ( C + (q-q')*alpha ) = (-0.6 + (-5)*(3-7)*0.01)x + (4+(3-7)*0.01) y= -0.4x+3.96
