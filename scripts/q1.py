import gym
import pybulletgym.envs
import time
import numpy as np
import pdb
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


l0 = .1
l1 = .11
alpha =.01

def l2(p1,p2):
	delta = p2-p1
	return(np.sqrt(np.sum(np.multiply(delta,delta))))

def normAng(theta):
	# if theta < 0.:
	# 	theta = theta + ((theta*-1//(np.pi*2))+1)*np.pi*2

	# theta = theta % (np.pi*2)
	# if theta > np.pi:
	# 	theta = theta - (np.pi*2)
	return theta

def c0(q0):
	return(np.array([[np.cos(q0), -np.sin(q0), l0*np.cos(q0)],
					[np.sin(q0), np.cos(q0), l0*np.sin(q0)],
					[0, 0, 1]]))

def c1(q1):
	return(np.array([[np.cos(q1), -np.sin(q1), l1*np.cos(q1)],
					[np.sin(q1), np.cos(q1), l1*np.sin(q1)],
					[0, 0, 1]]))

def getForwardModel(q0,q1):
	#return(np.dot(np.dot(c0(q0),c1(q1)),np.matrix([0,0,1]).T))
	return(np.array([[l1*np.cos(q0+q1) + l0*np.cos(q0)],
					[l1*np.sin(q0+q1) + l0*np.sin(q0)]]))
def getJacobian(q0,q1):
	return(np.array([[-l1*np.sin(q0+q1)-l0*np.sin(q0), -l1*np.sin(q1+q0)],
					[l1*np.cos(q0+q1)+l0*np.cos(q0), l1*np.cos(q0+q1)]]))

def getIK(q0,q1,target):
	cur_end_eff = getForwardModel(q0,q1)[:2,:]
	i = 0
	while l2(target,cur_end_eff) > .0001:
		vec = target - cur_end_eff
		jac = getJacobian(q0,q1)
		jdag = np.linalg.pinv(jac)
		offset = np.dot(jdag,vec)
		q0 += offset[0,0] * alpha
		q1 += offset[1,0] * alpha
		cur_end_eff = getForwardModel(q0,q1)[:2,:]
		i+=1
	print("solved in %d steps" % i)

	return(normAng(q0) ,normAng(q1))

def trajectory(t,T):
	theta = np.pi*2/T*t - np.pi
	return(np.array([[(0.19 + 0.02*np.cos(4*theta))*np.cos(theta)],
					[(0.19 + 0.02*np.cos(4*theta))*np.sin(theta)]]))


def FkKernel(q0,q1):
	m0 = np.dot(c0(q0),np.matrix([0,0,1]).T)
	m1 = np.dot(c1(q1), m0)
	return(np.vstack((m0[:2],m1[:2])))

def FastronClustering():
	N = 1000
	numClusters = 3
	colorarr = ['b','g','r','c','m','y','k','w']
	data = np.random.rand(N,2)
	data *= np.pi*2
	FkData = np.zeros((N,4))
	for i in range(N):
		FkData[i,:] = FkKernel(data[i,0],data[i,1]).T

	# kmeans = KMeans(n_clusters=4, random_state=0).fit(FkData)
	kmeans = KMeans(n_clusters=numClusters).fit(FkData)
	datasets = []
	for i in range(numClusters):
		index = [kmeans.labels_ == i]
		d = data[tuple(index)]
		datasets.append(d)
		plt.scatter(d[:,0],d[:,1],c=colorarr[i])


	
	# index = [kmeans.labels_ == 1]
	# d1 = data[index]
	# index = [kmeans.labels_ == 2]
	# d2 = data[index]
	# index = [kmeans.labels_ == 3]
	# d3 = data[index]
	# print(d0.shape)
	# print(kmeans.labels_)
	# plt.scatter(d0[:,0],d0[:,1],c='b')
	# plt.scatter(d1[:,0],d1[:,1],c='r')
	# plt.scatter(d2[:,0],d2[:,1],c='g')
	# plt.scatter(d3[:,0],d3[:,1],c='y')
	plt.show()





def endEffSpacePD():
	env = gym.make("ReacherPyBulletEnv-v0")
	#env = gym.make("HalfCheetahPyBulletEnv-v0")
	env.render(mode="rgbarray")
	env.reset()
	#pdb.set_trace()
	#env.unwrapped.robot.central_joint.reset_position(1,0)
	#print(env.unwrapped.robot.central_joint.current_position())
	done = False
	t = 0
	start = trajectory(t,300)
	actual_traj = np.zeros((2,300))
	desired_traj = np.zeros((2,300))
	robot_traj = np.zeros((2,300))

	q0s = []
	q1s = []
	q0ts = []
	q1ts = []
	tip_dist = []

	error_sum = 0

	actual_traj[:,t] = start.flatten()
	q0 = env.unwrapped.robot.central_joint.current_position()[0]
	q1 = env.unwrapped.robot.elbow_joint.current_position()[0]
	q0,q1 = getIK(q0,q1,start)
	env.unwrapped.robot.central_joint.reset_position(q0,0)
	env.unwrapped.robot.elbow_joint.reset_position(q1,0)
	robot_traj[:,t] = getForwardModel(q0,q1)[:2,:].flatten()
	desired_traj[:,t] = getForwardModel(q0,q1)[:2,:].flatten()
	p_prev = np.zeros((2,1))
	for i in range(300):
		print(i)
		time.sleep(0.05)
		target = trajectory(t,300)
		error_sum += l2(target,getForwardModel(q0,q1)[:2,:])**2
		for _ in range(100):
			p = target - getForwardModel(q0,q1)
			d = p - p_prev
			tor = p*.002 + d*.001
			p_prev = p
			jac = getJacobian(q0,q1)
			jdag = np.linalg.pinv(jac)
			tor = np.dot(jdag,tor)
			obs, reward, done, info = env.step((tor[0,0],tor[1,0]))
			env.render()
			q0 = env.unwrapped.robot.central_joint.current_position()[0]
			q1 = env.unwrapped.robot.elbow_joint.current_position()[0]
			q0s.append(q0)
			q1s.append(q1)
			robot_traj[:,t] = getForwardModel(q0,q1)[:2,:].flatten()
			dist = l2(getForwardModel(q0,q1)[:2,:],np.array([[0],[0]]))
			tip_dist.append(dist)


		actual_traj[:,t] = target.flatten()
		#desired_traj[:,t] = getForwardModel(q0t,q1t)[:2,:].flatten()

		#print(env.unwrapped.robot.central_joint.current_position(), t)
		t += 1
	#env.render()
	# time.sleep(1)

	env.close()
	plt.plot(actual_traj[0,:],actual_traj[1,:],c='b')
	plt.scatter(robot_traj[0,:],robot_traj[1,:],c=np.arange(0,300,1))
	plt.colorbar()
	#plt.plot(desired_traj[0,:],desired_traj[1,:],c='g')
	plt.show()

	# plt.plot(q0s,c='b')
	# plt.plot(q1s,c='r')
	# plt.plot(q0ts, c='g')
	# plt.plot(q1ts, c='y')
	# #plt.plot(tip_dist, c='k')
	# plt.show()

	# print(getForwardModel(np.pi/4*3, np.pi/2))
	# print(getJacobian(np.pi/4*3, np.pi/2))
	# q0,q1 = getIK(np.pi/4,0,np.array([[0],[np.sqrt(2)]]))
	# print(q0,q1)
	# print(getForwardModel(q0,q1))

	return(error_sum/300)





def jointSpacePD():
	env = gym.make("ReacherPyBulletEnv-v0")
	#env = gym.make("HalfCheetahPyBulletEnv-v0")
	env.render(mode="rgbarray")
	env.reset()
	#pdb.set_trace()
	#env.unwrapped.robot.central_joint.reset_position(1,0)
	#print(env.unwrapped.robot.central_joint.current_position())
	done = False
	t = 0
	start = trajectory(t,300)
	actual_traj = np.zeros((2,300))
	desired_traj = np.zeros((2,300))
	robot_traj = np.zeros((2,300))

	q0s = []
	q1s = []
	q0ts = []
	q1ts = []
	tip_dist = []

	error_sum = 0

	actual_traj[:,t] = start.flatten()
	q0 = env.unwrapped.robot.central_joint.current_position()[0]
	q1 = env.unwrapped.robot.elbow_joint.current_position()[0]
	q0,q1 = getIK(q0,q1,start)
	env.unwrapped.robot.central_joint.reset_position(q0,0)
	env.unwrapped.robot.elbow_joint.reset_position(q1,0)
	robot_traj[:,t] = getForwardModel(q0,q1)[:2,:].flatten()
	desired_traj[:,t] = getForwardModel(q0,q1)[:2,:].flatten()
	p_prev = np.zeros((2,1))
	for i in range(300):
		time.sleep(0.05)
		target = trajectory(t,300)
		error_sum += l2(target,getForwardModel(q0,q1)[:2,:])**2
		q0t, q1t = getIK(q0,q1,target)
		for _ in range(100):
			p = np.array([[q0t - q0],[q1t - q1]])
			d = p - p_prev
			tor = p*.002 + d*.001
			p_prev = p
			obs, reward, done, info = env.step((tor[0,0],tor[1,0]))
			env.render()
			q0 = env.unwrapped.robot.central_joint.current_position()[0]
			q1 = env.unwrapped.robot.elbow_joint.current_position()[0]
			q0s.append(q0)
			q1s.append(q1)
			q0ts.append(q0t)
			q1ts.append(q1t)
			robot_traj[:,t] = getForwardModel(q0,q1)[:2,:].flatten()
			dist = l2(getForwardModel(q0,q1)[:2,:],np.array([[0],[0]]))
			tip_dist.append(dist)


		actual_traj[:,t] = target.flatten()
		desired_traj[:,t] = getForwardModel(q0t,q1t)[:2,:].flatten()

		#print(env.unwrapped.robot.central_joint.current_position(), t)
		t += 1
	#env.render()
	# time.sleep(1)

	env.close()
	plt.plot(actual_traj[0,:],actual_traj[1,:],c='b')
	plt.scatter(robot_traj[0,:],robot_traj[1,:],c=np.arange(0,300,1))
	plt.colorbar()
	plt.plot(desired_traj[0,:],desired_traj[1,:],c='g')
	plt.show()

	# plt.plot(q0s,c='b')
	# plt.plot(q1s,c='r')
	# plt.plot(q0ts, c='g')
	# plt.plot(q1ts, c='y')
	# #plt.plot(tip_dist, c='k')
	# plt.show()

	# print(getForwardModel(np.pi/4*3, np.pi/2))
	# print(getJacobian(np.pi/4*3, np.pi/2))
	# q0,q1 = getIK(np.pi/4,0,np.array([[0],[np.sqrt(2)]]))
	# print(q0,q1)
	# print(getForwardModel(q0,q1))

	return(error_sum/300)


if __name__ == "__main__":
	FastronClustering()
	# print(endEffSpacePD())