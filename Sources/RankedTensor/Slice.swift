//
//  Slice.swift
//  RankedTensor
//
//  Copyright 2016-2017 The DLVM Team.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//

import struct CoreTensor.TensorIndex
import struct CoreTensor.TensorShape
import struct CoreTensor.TensorSlice

/// Ranked tensor slice
public struct RankedTensorSlice<Rank: StaticRank> : RankedTensorProtocol {
    public typealias UnitType = Rank.UnitType
    public typealias Shape = Rank.Shape
    public typealias BaseForm = Tensor<Rank>
    public typealias ElementTensor = Rank.ElementTensor

    internal var base: TensorSlice<UnitType>

    private init(base: TensorSlice<UnitType>) {
        self.base = base
    }
}

public extension RankedTensorSlice {
    /// Tensor rank
    var rank: UInt {
        return Rank.rank
    }

    /// Tensor shape
    var shape: Shape {
        return Rank.staticShape(from: base.shape)
    }

    var elementShape: TensorShape? {
        return base.elementShape
    }

    /// The number of items (atom) in the tensor.
    var unitCount: Int {
        return units.count
    }

    /// Capacity reserved for element tensors
    var capacity: Int {
        return units.capacity / unitCountPerElement
    }

    var unitCountPerElement: Int {
        return base.unitCountPerElement
    }

    var units: ArraySlice<UnitType> {
        return base.units
    }

    var indices: CountableRange<Int> {
        return base.indices
    }

    var startIndex: Int {
        return base.startIndex
    }

    var endIndex: Int {
        return base.endIndex
    }
}

public extension RankedTensorSlice where Shape == Shape1D {
    /// Initialize a vector from units
    init<C: Collection>(_ units: C) where C.Element == UnitType, C.IndexDistance == Int {
        self.init(shape: (UInt(units.count)), units: units)
    }
}

extension RankedTensorSlice {
    init<S>(base: Tensor<S>, indices: [Int]) where S.UnitType == UnitType {
        self.init(base: TensorSlice(base: base.base, indices: indices))
    }

    init<S>(base: RankedTensorSlice<S>, indices: [Int]) where S.UnitType == UnitType {
        self.init(base: TensorSlice(base: base.base, indices: indices))
    }
}

public extension RankedTensorSlice {
    init(_ base: Tensor<Rank>) {
        self.init(base: TensorSlice(base.base))
    }

    init(base: Tensor<Rank>, bounds: CountableRange<Int>?) {
        self.init(base: TensorSlice(base: base.base, bounds: bounds))
    }

    init(base: RankedTensorSlice, bounds: CountableRange<Int>?) {
        self.init(base: TensorSlice(base: base.base, bounds: bounds))
    }

    init<S>(base: Tensor<S>, index: Int)
        where S.ElementTensor == RankedTensorSlice, S.UnitType == UnitType
    {
        self.init(base: TensorSlice(base: base.base, index: index))
    }

    init<S>(base: RankedTensorSlice<S>, index: Int)
        where S.ElementTensor == RankedTensorSlice, S.UnitType == UnitType
    {
        self.init(base: TensorSlice(base: base.base, index: index))
    }
}

public extension RankedTensorSlice {
    /// Initialize a tensor using an existing slice of elements in row-major order
    /// - parameter shape: tensor shape
    /// - parameter elements: slice of existing elements in row-major order
    internal init(shape: Rank.Shape, units: ContiguousArray<UnitType>) {
        self.init(Tensor(shape: shape, units: units))
    }

    /// Allocate and initialize a tensor to a repeated value
    /// - parameter shape: tensor shape
    /// - parameter repeating: repeated value
    init(shape: Rank.Shape, repeating repeatedValue: UnitType) {
        self.init(Tensor(shape: shape, repeating: repeatedValue))
    }

    /// Allocate and initialize a tensor using the factory function
    /// - parameter shape: tensor shape
    /// - parameter supplier: factory function providing values lazily
    init(shape: Rank.Shape, supplier: () -> UnitType) {
        self.init(Tensor(shape: shape, supplier: supplier))
    }

    /// Initialize a tensor from a sequence of elements in row-major order
    /// - parameter shape: tensor shape
    /// - parameter elements: sequence of elements in row-major order
    /// - parameter vacancySupplier
    init<S: Sequence>(shape: Shape, units: S,
                      vacancySupplier supplier: (() -> UnitType)? = nil)
        where S.Element == UnitType {
            self.init(Tensor(shape: shape, units: units, vacancySupplier: supplier))
    }
}

public extension RankedTensorSlice {
    var dynamicShape: TensorShape {
        return base.shape
    }
}

public extension RankedTensorSlice {
    func isSimilar<A>(to other: RankedTensorSlice<A>) -> Bool where A.UnitType == UnitType {
        return base.isSimilar(to: other.base)
    }

    func isIsomorphic<A>(to other: RankedTensorSlice<A>) -> Bool where A.UnitType == UnitType {
        return base.isIsomorphic(to: other.base)
    }
}

public extension RankedTensorSlice where Rank.UnitType : Strideable {
    init(shape: Shape, unitsIncreasingFrom lowerBound: UnitType) {
        self.init(Tensor(shape: shape, unitsIncreasingFrom: lowerBound))
    }
}

public extension RankedTensorSlice where Rank.UnitType : Strideable, Rank.UnitType.Stride : SignedInteger, Rank.Shape == (UInt) {
    init(scalarElementsIn bounds: CountableRange<UnitType>) {
        self.init(Tensor(scalarElementsIn: bounds))
    }

    init(scalarElementsIn bounds: CountableClosedRange<UnitType>) {
        self.init(Tensor(scalarElementsIn: bounds))
    }
}

public extension RankedTensorSlice {
    mutating func updateUnit(at index: Int, to newValue: UnitType) {
        base.updateUnit(at: index, to: newValue)
    }
}

public extension RankedTensorSlice where Rank.UnitType : Numeric {
    mutating func incrementUnit(at index: Int, by newValue: UnitType) {
        base.incrementUnit(at: index, by: newValue)
    }

    mutating func decrementUnit(at index: Int, by newValue: UnitType) {
        base.decrementUnit(at: index, by: newValue)
    }

    mutating func multiplyUnit(at index: Int, by newValue: UnitType) {
        base.multiplyUnit(at: index, by: newValue)
    }
}

public extension RankedTensorSlice where Rank.UnitType : BinaryInteger {
    mutating func divideUnit(at index: Int, by newValue: UnitType) {
        base.divideUnit(at: index, by: newValue)
    }
}

public extension RankedTensorSlice where Rank.UnitType : FloatingPoint {
    mutating func divideUnit(at index: Int, by newValue: UnitType) {
        base.divideUnit(at: index, by: newValue)
    }
}

extension RankedTensorSlice : RandomAccessCollection {
    public typealias Index = Int
    public typealias Element = ElementTensor
    public typealias SubSequence = RankedTensorSlice<Rank>

    /// Access the scalar element or element tensor at index
    public subscript(index: Int) -> Element {
        get {
            return Rank.element(of: self, at: index)
        }
        set {
            Rank.updateElement(newValue, at: index, in: &self)
        }
    }

    /// Access the subtensor specified by a contiguous range of indices
    public subscript(bounds: Range<Int>) -> SubSequence {
        get {
            precondition(indices ~= bounds.lowerBound && indices ~= bounds.upperBound - 1,
                         "Slice indices are out of bounds")
            return SubSequence(base: self, bounds: CountableRange(bounds))
        }
        set {
            precondition(indices ~= bounds.lowerBound && indices ~= bounds.upperBound - 1,
                         "Slice indices are out of bounds")
            precondition(newValue.dynamicShape == dynamicShape.dropFirst().prepending(bounds.count),
                         "Shape mismatch")
            base[bounds] = newValue.base
        }
    }
}

public extension RankedTensorSlice {
    func withUnsafeBufferPointer<Result>
        (_ body: (UnsafeBufferPointer<UnitType>) throws -> Result) rethrows -> Result {
        return try base.withUnsafeBufferPointer(body)
    }

    mutating func withUnsafeMutableBufferPointer<Result>
        (_ body: (inout UnsafeMutableBufferPointer<UnitType>) throws -> Result) rethrows -> Result {
        return try base.withUnsafeMutableBufferPointer(body)
    }
}

extension RankedTensorSlice : TextOutputStreamable {
    public func write<Target>(to target: inout Target) where Target : TextOutputStream {
        return base.write(to: &target)
    }
}

public typealias TensorSlice1D<T> = RankedTensorSlice<Rank1<T>>
public typealias TensorSlice2D<T> = RankedTensorSlice<Rank2<T>>
public typealias TensorSlice3D<T> = RankedTensorSlice<Rank3<T>>
public typealias TensorSlice4D<T> = RankedTensorSlice<Rank4<T>>

public typealias VectorSlice<T> = TensorSlice1D<T>
public typealias MatrixSlice<T> = TensorSlice2D<T>

